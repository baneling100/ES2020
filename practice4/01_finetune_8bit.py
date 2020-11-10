'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

base_path = "./"

import sys
sys.path.append(base_path)

from train_test import train, test

LR = 0.01
EPOCH = 5

# Data
print('==> Preparing data')
from dataset import cifar10_dataset
trainloader, testloader = cifar10_dataset(base_path + "/data")

# Model
print('==> Building model')
from resnet_quant import ResNet18
model = ResNet18()
model.load_state_dict(torch.load(base_path + "/train_best.pth"))

if torch.cuda.is_available():
    model.cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
criterion = torch.nn.CrossEntropyLoss()

print('==> Full-precision model accuracy')
from quant_op import Q_ReLU, Q_Conv2d, Q_Linear
test(model, testloader, criterion)

for name, module in model.named_modules():
    if isinstance(module, Q_ReLU):
        module.n_lv = 8
        module.bound = 1
    
    if isinstance(module, (Q_Conv2d, Q_Linear)):
        module.n_lv = 8
        module.ratio = 0.5

print('==> Quantized model accuracy')
from quant_op import Q_ReLU, Q_Conv2d, Q_Linear
test(model, testloader, criterion)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, last_epoch=start_epoch-1)

for epoch in range(start_epoch, start_epoch + EPOCH):
    scheduler.step()
    train(model, trainloader, criterion, optimizer, epoch)
    acc = test(model, testloader, criterion)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.module.state_dict(), base_path +  "quant_best.pth")

print('==> Fine-tuned model accuracy')
from quant_op import Q_ReLU, Q_Conv2d, Q_Linear
test(model, testloader, criterion)

