from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=90, type=int, help='training target epoch')
args = parser.parse_args()

# Data
print('==> Preparing data..')
from dataset import cifar10_dataset
trainloader, testloader = cifar10_dataset("./data")

# Model
print('==> Building model..')
from resnet import ResNet18
model = ResNet18()

if torch.cuda.is_available():
    model.cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, last_epoch=start_epoch-1)

from train_test import train, test
for epoch in range(start_epoch, start_epoch + args.epoch):
    scheduler.step()
    train(model, trainloader, criterion, optimizer, epoch)
    acc = test(model, testloader, criterion)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "./train_best.pth")