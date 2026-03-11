"""Evaluate OTAD under BPDA+PGD attack (Linf + L2)."""

import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os

from BPDA import BPDAWrapper, PGDAttack
from models.models import CNNBlock, DMLResNet
from models.vit import ViT, ViT_feat
from models.cipnet import CIPNet
from solvers.mosek_potential import LP
from solvers.mosek_test import QCQP

parser = argparse.ArgumentParser()
parser.add_argument('--defense', type=str, default='otad-t', choices=['otad-t', 'otad-t-nn'])
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
subset_indices = range(1000)
test_subset = torch.utils.data.Subset(testset, subset_indices)
testloader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=True)

num_s = 10
L1 = 2
l1 = 0
L2 = 2
l2 = 0

def find_neighbors_batch(data, targets, k):
    """Find k nearest neighbors for a batch of target vectors."""
    distances = torch.cdist(targets, data)
    _, indices = torch.topk(distances, k, largest=False, dim=1)
    return indices

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

midnet = ViT_feat(3, 10, img_size=32, patch=8, dropout=0.0, mlp_hidden=384,
                  num_layers=7, hidden=384, head=12, is_cls_token=True).cuda()
midnet.load_state_dict(new_state_dict)
midnet.eval()
for param in midnet.parameters():
    param.requires_grad = False

# Load precomputed OT data
OTrawdata = torch.from_numpy(np.load('./precomputed/rawdata.npy'))
OTinput = np.load('./precomputed/otinput.npy')
OToutput = np.load('./precomputed/otoutput.npy')

# Load DML ResNet
dmlnet = DMLResNet(CNNBlock, [2, 2, 2]).cuda()
dmlstate_dict = torch.load('./checkpoints/dml_resnet.pth', weights_only=True)
dmlnet.load_state_dict(dmlstate_dict)
dmlnet.eval()
for param in dmlnet.parameters():
    param.requires_grad = False

# Build DML feature database
newlabels_np = np.load('./precomputed/labels.npy')
newlabels = torch.from_numpy(newlabels_np)

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

newtrainset = MyDataset(OTrawdata, newlabels)
newtrainloader = torch.utils.data.DataLoader(newtrainset, batch_size=128, shuffle=False)
for batch_idx, (inputs, targets) in enumerate(newtrainloader):
    inputs = inputs.cuda().view(inputs.shape[0], 3, 32, 32)
    outputs = dmlnet(inputs)
    if batch_idx == 0:
        dmlfeature = outputs.detach()
    else:
        dmlfeature = torch.cat((dmlfeature, outputs.detach()), dim=0)
print(f'DML feature shape: {dmlfeature.shape}')

# OTAD-T defense (uses MOSEK solvers)
class OTAD(nn.Module):
    def __init__(self, net):
        super(OTAD, self).__init__()
        self.net = net

    def forward(self, x):
        lt = l1
        Lt = L1
        cls_token = dmlnet(x).detach()
        embed = net.embedding(net.normalization(x)).view(x.shape[0], -1).detach().cpu().numpy()
        indices = find_neighbors_batch(dmlfeature, cls_token, num_s).cpu().numpy()
        input_local = OTinput[indices].squeeze(0)
        output_local = OToutput[indices].squeeze(0)
        while 1:
            try:
                U = LP(lt, Lt, input_local, output_local)
            except:
                Lt = Lt + 1
                lt = lt - 1
                continue
            else:
                break
        v, test_output = QCQP(l2, L2, input_local, output_local, U, embed)
        test_output = torch.from_numpy(test_output).cuda().float()
        return test_output.view(x.shape[0], 65, 384)

# OTAD-T-NN defense (uses QCQPNet)
if args.defense == 'otad-t-nn':
    cipnet = CIPNet(num_neighbors=10*2, point_dim=24960, dim=2048, depth=6,
                    heads=8, mlp_dim=512, dropout=0.1).cuda()
    cip_state_dict = torch.load('./checkpoints/cipnet.pth', weights_only=True)
    cipnet.load_state_dict(cip_state_dict)
    cipnet.eval()
    for param in cipnet.parameters():
        param.requires_grad = False

    OTinput_tensor = torch.from_numpy(OTinput).cuda()
    OToutput_tensor = torch.from_numpy(OToutput).cuda()

class OTAD_NN(nn.Module):
    def __init__(self, net):
        super(OTAD_NN, self).__init__()
        self.net = net

    def forward(self, x):
        cls_token = dmlnet(x).detach()
        indices = find_neighbors_batch(dmlfeature, cls_token, num_s)
        input_neighbors = OTinput_tensor[indices]
        output_neighbors = OToutput_tensor[indices]
        embed = net.embedding(net.normalization(x)).view(x.shape[0], -1)
        neighbors = torch.cat((input_neighbors, output_neighbors), 1).cuda()
        test_output = cipnet(embed, neighbors)
        return test_output.view(x.shape[0], 65, 384)

class ClassifierWrapper(nn.Module):
    def __init__(self, classifier):
        super(ClassifierWrapper, self).__init__()
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(x)

wrapped_classifier = ClassifierWrapper(net.classifier)

if args.defense == 'otad-t':
    defense = nn.Sequential(OTAD(net))
else:
    defense = nn.Sequential(OTAD_NN(net))

defense_withbpda = BPDAWrapper(defense, forwardsub=midnet)
defended_model = nn.Sequential(defense_withbpda, wrapped_classifier)
defended_model.eval()

# --- Linf PGD Attack ---
total = 0
correct_1 = 0
correct_2 = 0
correct_3 = 0

print(f'\n==> Linf PGD attack ({args.defense})...')
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.cuda(), targets.cuda()

    atk = PGDAttack(defended_model, eps=8/255, eps_iter=2/255, nb_iter=20)
    inputs_adv = atk.perturb(inputs, targets)

    outputs_1 = net(inputs_adv)
    _, predicted_1 = outputs_1.max(1)
    correct_1 += predicted_1.eq(targets).sum().item()

    outputs_2 = net.classifier(defense(inputs_adv))
    _, predicted_2 = outputs_2.max(1)
    correct_2 += predicted_2.eq(targets).sum().item()

    outputs_3 = net.classifier(defense(inputs))
    _, predicted_3 = outputs_3.max(1)
    correct_3 += predicted_3.eq(targets).sum().item()

    total += targets.size(0)

    if (batch_idx + 1) % 100 == 0:
        print(f'[{total}] No defense: {100.*correct_1/total:.2f}% | '
              f'Defense (adv): {100.*correct_2/total:.2f}% | '
              f'Defense (clean): {100.*correct_3/total:.2f}%')

# --- L2 PGD Attack ---
total = 0
correct_2 = 0
correct_3 = 0

print(f'\n==> L2 PGD attack ({args.defense})...')
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.cuda(), targets.cuda()

    atk = PGDAttack(defended_model, eps=0.5, eps_iter=0.05, nb_iter=20, ord=2)
    inputs_adv = atk.perturb(inputs, targets)

    outputs_2 = net.classifier(defense(inputs_adv))
    _, predicted_2 = outputs_2.max(1)
    correct_2 += predicted_2.eq(targets).sum().item()

    outputs_3 = net.classifier(defense(inputs))
    _, predicted_3 = outputs_3.max(1)
    correct_3 += predicted_3.eq(targets).sum().item()

    total += targets.size(0)

    if (batch_idx + 1) % 100 == 0:
        print(f'[{total}] Defense (adv): {100.*correct_2/total:.2f}% | '
              f'Defense (clean): {100.*correct_3/total:.2f}%')
