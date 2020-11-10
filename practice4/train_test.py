from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import torch
_print_freq = 10


def set_print_freq(freq):
    global _print_freq
    _print_freq = freq


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def test(model, val_loader, criterion, train=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.train(train)

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():        
            if not isinstance(model, torch.nn.DataParallel):
                input = input.cuda()
            target = target.cuda(non_blocking=True)        
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)      
        
        loss = criterion(output, target_var)
        prec1, = accuracy(output.data, target, topk=(1,))

        # record loss and accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i+1, len(val_loader), 
                    batch_time=batch_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1) )
    return top1.avg


def train(model, train_loader, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not isinstance(model, torch.nn.DataParallel):
            input = input.cuda()

        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0:
            print('Epoch {0}: [{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i+1, len(train_loader),
                    batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))
