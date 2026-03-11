"""Model building blocks and DML network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, planes):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(planes)
        self.lin1 = nn.Linear(planes,planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.lin2 = nn.Linear(planes,planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.lin1(out)
        out = self.lin2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class CNNBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(CNNBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class DMLResNet(nn.Module):
    """ResNet for deep metric learning, outputs L2-normalized 128-d embeddings."""
    def __init__(self, block, num_blocks):
        super(DMLResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2 = self._make_layer2(block, 64, num_blocks[0], stride=1)
        self.layer3 = self._make_layer2(block, 128, num_blocks[1], stride=2)
        self.layer4 = self._make_layer2(block, 256, num_blocks[2], stride=2)
        self.lin = nn.Linear(256,128)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        self.mean = torch.tensor(mean).view(3, 1, 1).cuda()
        self.std = torch.tensor(std).view(3, 1, 1).cuda()

    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.conv1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)
        out = self.lin(out)
        # L2 normalize
        out = out / out.norm(p=2, dim=1, keepdim=True)
        return out
