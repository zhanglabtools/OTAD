"""Train DML ResNet for neighbor retrieval."""

import random
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import os

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from models.models import CNNBlock, DMLResNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 42
set_random_seed(seed)

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
subset_indices = range(1000)
test_subset = torch.utils.data.Subset(testset, subset_indices)
testloader = torch.utils.data.DataLoader(test_subset, batch_size=100, shuffle=True)

net = DMLResNet(CNNBlock, [2, 2, 2]).cuda()

optimizer = optim.Adam(net.parameters(), lr=0.001)
num_epochs = 200

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.1)

reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, reducer=reducer)
mining_func = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

def train(model, loss_func, mining_func, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} Iter {batch_idx}: Loss = {loss:.4f}, "
                  f"Triplets = {mining_func.num_triplets}")

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print(f"Test set accuracy (Precision@1) = {accuracies['precision_at_1']}")

for epoch in range(1, num_epochs + 1):
    train(net, loss_func, mining_func, trainloader, optimizer, epoch)
    test(trainset, test_subset, net, accuracy_calculator)
    scheduler.step()

os.makedirs('./checkpoints', exist_ok=True)
state = net.state_dict()
torch.save(state, './checkpoints/dml_resnet.pth')
print('Model saved to ./checkpoints/dml_resnet.pth')
