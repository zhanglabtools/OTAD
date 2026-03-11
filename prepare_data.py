"""Prepare OT data (embeddings and encoder outputs) from the ViT backbone."""

import random
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

from models.vit import ViT

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

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

print('==> Extracting OT data...')
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.cuda()
    embedding = net.embedding(net.normalization(inputs))
    re_outputs = net.enc(embedding).view(inputs.shape[0], -1)
    inputs_flat = inputs.view(inputs.shape[0], -1)

    otrawdata_np = inputs_flat.detach().cpu().numpy()
    if batch_idx == 0:
        otrawdata = otrawdata_np
    else:
        otrawdata = np.vstack((otrawdata, otrawdata_np))

    labels_np = targets.view(inputs.shape[0], -1).numpy()
    if batch_idx == 0:
        labels = labels_np
    else:
        labels = np.vstack((labels, labels_np))

    otinputs_np = embedding.view(inputs.shape[0], -1).detach().cpu().numpy()
    if batch_idx == 0:
        otinput = otinputs_np
    else:
        otinput = np.vstack((otinput, otinputs_np))

    otoutputs_np = re_outputs.detach().cpu().numpy()
    if batch_idx == 0:
        otoutput = otoutputs_np
    else:
        otoutput = np.vstack((otoutput, otoutputs_np))

print(f'labels: {labels.shape}')
print(f'rawdata: {otrawdata.shape}')
print(f'otinput: {otinput.shape}')
print(f'otoutput: {otoutput.shape}')

os.makedirs('./precomputed', exist_ok=True)
np.save('./precomputed/labels.npy', labels)
np.save('./precomputed/rawdata.npy', otrawdata)
np.save('./precomputed/otinput.npy', otinput)
np.save('./precomputed/otoutput.npy', otoutput)
print('Done.')
