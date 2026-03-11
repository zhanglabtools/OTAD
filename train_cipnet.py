"""Train CIPNet to approximate CIP solutions."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os

from models.cipnet import CIPNet
from models.vit import ViT

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class CIPDataset(Dataset):
    def __init__(self, embed_input, cip_output, neighbor_indices):
        self.embed_input = embed_input.squeeze(1)
        self.cip_output = cip_output.squeeze(1)
        self.neighbor_indices = neighbor_indices

    def __len__(self):
        return len(self.embed_input)

    def __getitem__(self, idx):
        base_point = self.embed_input[idx]
        cip_sol = self.cip_output[idx]
        neighbor_idx = self.neighbor_indices[idx]
        return base_point, neighbor_idx, cip_sol

def load_data(data_paths):
    embed_input = torch.from_numpy(np.load(data_paths[0]))
    cip_output = torch.from_numpy(np.load(data_paths[1]))
    neighbor_indices = np.load(data_paths[2])
    dataset = CIPDataset(embed_input, cip_output, neighbor_indices)
    return dataset

# Load precomputed OT data
OTinput = np.load('./precomputed/otinput.npy')
OToutput = np.load('./precomputed/otoutput.npy')

data_paths = [
    './precomputed/cip_embedinput.npy',
    './precomputed/cip_output.npy',
    './precomputed/cip_neighbors.npy'
]

dataset = load_data(data_paths)

train_size = int(0.8 * len(dataset))
print(f'Train size: {train_size}')
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

net = CIPNet(num_neighbors=10*2, point_dim=24960, dim=2048, depth=6,
             heads=8, mlp_dim=512, dropout=0.1).cuda()

# Load ViT for optional regularization
state_dict = torch.load('./checkpoints/vit_cifar10.pth', weights_only=True)
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('model.', '')
    new_state_dict[new_key] = value

midnet = ViT(3, 10, img_size=32, patch=8, dropout=0.0, mlp_hidden=384,
             num_layers=7, hidden=384, head=12, is_cls_token=True).cuda()
midnet.load_state_dict(new_state_dict)
midnet.eval()
for param in midnet.parameters():
    param.requires_grad = False

optimizer = optim.Adam(net.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
criterion = nn.MSELoss()
alpha = 1.0

def train(epoch):
    print(f'\nEpoch: {epoch}')
    train_loss = 0
    net.train()
    for batch_idx, (base_points, neighbor_idx, cip_sols) in enumerate(train_loader):
        input_neighbors = torch.from_numpy(OTinput[neighbor_idx])
        output_neighbors = torch.from_numpy(OToutput[neighbor_idx])
        neighbors = torch.cat((input_neighbors, output_neighbors), 1).cuda()
        base_points, cip_sols = base_points.cuda(), cip_sols.cuda()
        optimizer.zero_grad()
        outputs = net(base_points, neighbors)
        loss = criterion(outputs, cip_sols)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Train Loss: {train_loss/(batch_idx+1):.3f}')
    return train_loss / (batch_idx + 1)

def test(epoch):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (base_points, neighbor_idx, cip_sols) in enumerate(test_loader):
            input_neighbors = torch.from_numpy(OTinput[neighbor_idx])
            output_neighbors = torch.from_numpy(OToutput[neighbor_idx])
            neighbors = torch.cat((input_neighbors, output_neighbors), 1).cuda()
            base_points, cip_sols = base_points.cuda(), cip_sols.cuda()
            outputs = net(base_points, neighbors)
            loss = criterion(outputs, cip_sols)
            test_loss += loss.item()
        print(f'Test Loss: {test_loss/(batch_idx+1):.3f}')
    return test_loss / (batch_idx + 1)

for epoch in range(350):
    train(epoch)
    scheduler.step()
    test(epoch)
    state = net.state_dict()
    torch.save(state, './checkpoints/cipnet.pth')

print('Training complete. Model saved to ./checkpoints/cipnet.pth')
