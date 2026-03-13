"""Evaluate OTAD-T-NN under AutoAttack (Linf + L2)."""

import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

from autoattack import AutoAttack
from models.models import CNNBlock, DMLResNet
from models.vit import ViT
from models.cipnet import CIPNet

parser = argparse.ArgumentParser()
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
testloader = torch.utils.data.DataLoader(test_subset, batch_size=100, shuffle=True)

num_s = 10

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

# Load precomputed OT data
OTrawdata = np.load('./precomputed/rawdata.npy')
OTinput_np = np.load('./precomputed/otinput.npy')
OToutput_np = np.load('./precomputed/otoutput.npy')

# Load DML ResNet
dmlnet = DMLResNet(CNNBlock, [2, 2, 2]).cuda()
dmlstate_dict = torch.load('./checkpoints/dml_resnet.pth', weights_only=True)
dmlnet.load_state_dict(dmlstate_dict)
dmlnet.eval()
for param in dmlnet.parameters():
    param.requires_grad = False

# Build DML feature database
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

newtrainset = MyDataset(newdatas, newlabels)
newtrainloader = torch.utils.data.DataLoader(newtrainset, batch_size=128, shuffle=False)
for batch_idx, (inputs, targets) in enumerate(newtrainloader):
    inputs = inputs.cuda().view(inputs.shape[0], 3, 32, 32)
    outputs = dmlnet(inputs)
    if batch_idx == 0:
        dmlfeature = outputs.detach()
    else:
        dmlfeature = torch.cat((dmlfeature, outputs.detach()), dim=0)
print(f'DML feature shape: {dmlfeature.shape}')

# OTAD-T-NN defense
cipnet = CIPNet(num_neighbors=10*2, point_dim=24960, dim=2048, depth=6,
                heads=8, mlp_dim=512, dropout=0.1).cuda()
cip_state_dict = torch.load('./checkpoints/cipnet.pth', weights_only=True)
cipnet.load_state_dict(cip_state_dict)
cipnet.eval()
for param in cipnet.parameters():
    param.requires_grad = False

OTinput_tensor = torch.from_numpy(OTinput_np).cuda()
OToutput_tensor = torch.from_numpy(OToutput_np).cuda()

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
defense = nn.Sequential(OTAD_NN(net))
defended_model = nn.Sequential(defense, wrapped_classifier)
defended_model.eval()

# --- L2 AutoAttack ---
total = 0
correct_robust = 0
correct_standard = 0

print(f'\n==> L2 AutoAttack (otad-t-nn)...')
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.cuda(), targets.cuda()

    adversary = AutoAttack(defended_model, norm='L2', eps=0.5, version='standard', verbose=True)
    inputs_adv = adversary.run_standard_evaluation(inputs, targets)

    outputs_robust = net.classifier(defense(inputs_adv))
    _, predicted_robust = outputs_robust.max(1)
    correct_robust += predicted_robust.eq(targets).sum().item()

    outputs_standard = net.classifier(defense(inputs))
    _, predicted_standard = outputs_standard.max(1)
    correct_standard += predicted_standard.eq(targets).sum().item()

    total += targets.size(0)

    if (batch_idx + 1) % 100 == 0:
        print(f'[{total}] Robust Accuracy: {100.*correct_robust/total:.2f}% | '
              f'Standard Accuracy: {100.*correct_standard/total:.2f}%')

# --- Linf AutoAttack ---
total = 0
correct_robust = 0
correct_standard = 0

print(f'\n==> Linf AutoAttack (otad-t-nn)...')
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.cuda(), targets.cuda()

    adversary = AutoAttack(defended_model, norm='Linf', eps=8/255, version='standard', verbose=True)
    inputs_adv = adversary.run_standard_evaluation(inputs, targets)

    outputs_robust = net.classifier(defense(inputs_adv))
    _, predicted_robust = outputs_robust.max(1)
    correct_robust += predicted_robust.eq(targets).sum().item()

    outputs_standard = net.classifier(defense(inputs))
    _, predicted_standard = outputs_standard.max(1)
    correct_standard += predicted_standard.eq(targets).sum().item()

    total += targets.size(0)

    if (batch_idx + 1) % 100 == 0:
        print(f'[{total}] Robust Accuracy: {100.*correct_robust/total:.2f}% | '
              f'Standard Accuracy: {100.*correct_standard/total:.2f}%')
