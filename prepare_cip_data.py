"""Prepare CIP training data (solve LP + QCQP for each training sample)."""

import random
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

from models.models import CNNBlock, DMLResNet
from models.vit import ViT
from solvers.mosek_potential import LP
from solvers.mosek_test import QCQP

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 42
set_random_seed(seed)

print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
trainOTloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

num_s = 10
L1 = 2
l1 = 0
L2 = 2
l2 = 0

def distancevector_1(data, target, d):
    distance = data - target
    dvector = np.linalg.norm(distance, axis=1)
    dsort = np.argsort(dvector)
    return dsort[1:d], dvector

# Load ViT backbone
state_dict = torch.load('./checkpoints/vit_cifar10.pth', weights_only=True)
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('model.', '')
    new_state_dict[new_key] = value

net = ViT(3, 10, img_size=32, patch=8, dropout=0.0, mlp_hidden=384,
          num_layers=7, hidden=384, head=12, is_cls_token=True).cuda()
net.load_state_dict(new_state_dict)
net.eval()
for param in net.parameters():
    param.requires_grad = False

# Load DML ResNet
dmlnet = DMLResNet(CNNBlock, [2, 2, 2]).cuda()
dmlstate_dict = torch.load('./checkpoints/dml_resnet.pth', weights_only=True)
dmlnet.load_state_dict(dmlstate_dict)
dmlnet.eval()
for param in dmlnet.parameters():
    param.requires_grad = False

# Load precomputed OT data
OTrawdata = np.load('./precomputed/rawdata.npy')
OTinput = np.load('./precomputed/otinput.npy')
OToutput = np.load('./precomputed/otoutput.npy')
newlabels_np = np.load('./precomputed/labels.npy')

newdatas = torch.from_numpy(OTrawdata).float()
newlabels = torch.from_numpy(newlabels_np)
print(f'labels shape: {newlabels.shape}')

class MyDataset():
    def __init__(self, data, labels):
        imgs = []
        for i in range(len(labels)):
            imgs.append((data[i], labels[i]))
        self.imgs = imgs

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        return fn, label

    def __len__(self):
        return len(self.imgs)

# Compute DML features for all training data
newtrainset = MyDataset(newdatas, newlabels)
newtrainloader = torch.utils.data.DataLoader(newtrainset, batch_size=128, shuffle=False)
for batch_idx, (inputs, targets) in enumerate(newtrainloader):
    inputs = inputs.cuda().view(inputs.shape[0], 3, 32, 32)
    outputs = dmlnet(inputs)
    dmlfeature_np = outputs.detach().cpu().numpy()
    if batch_idx == 0:
        dmlfeature = dmlfeature_np
    else:
        dmlfeature = np.vstack((dmlfeature, dmlfeature_np))
print(f'dmlfeature shape: {dmlfeature.shape}')

# Solve LP + QCQP for each training sample
print('Making CIP training data...')
neighbors = []
embedinput = []
uu = []
cipout = []
vv = []

os.makedirs('./precomputed', exist_ok=True)

for batch_idx, (inputs, targets) in enumerate(trainOTloader):
    lt = l1
    Lt = L1
    inputs = inputs.cuda()
    cls_token = dmlnet(inputs).detach().cpu().numpy()
    embed = net.embedding(net.normalization(inputs)).view(inputs.shape[0], -1).detach().cpu().numpy()
    embedinput.append(embed)

    dsort, _ = distancevector_1(dmlfeature, cls_token, num_s + 1)
    neighbors.append(dsort)

    input_local = OTinput[dsort, :]
    output_local = OToutput[dsort, :]

    while 1:
        try:
            U = LP(lt, Lt, input_local, output_local)
        except:
            Lt = Lt + 1
            lt = lt - 1
            continue
        else:
            break

    uu.append(U)
    v, new_output = QCQP(l2, L2, input_local, output_local, U, embed)
    cipout.append(new_output.reshape(1, -1).astype(np.float32))
    vv.append(v.reshape(1, -1).astype(np.float32))

    if (batch_idx + 1) % 1000 == 0:
        print(f'Batch {batch_idx + 1} completed, saving...')
        np.save(f'./precomputed/cip_embedinput_{batch_idx+1}.npy', embedinput)
        np.save(f'./precomputed/cip_neighbors_{batch_idx+1}.npy', neighbors)
        np.save(f'./precomputed/cip_uu_{batch_idx+1}.npy', uu)
        np.save(f'./precomputed/cip_output_{batch_idx+1}.npy', cipout)
        np.save(f'./precomputed/cip_vv_{batch_idx+1}.npy', vv)
        embedinput.clear()
        neighbors.clear()
        uu.clear()
        cipout.clear()
        vv.clear()
        print(f'Batch {batch_idx + 1} data saved.')
