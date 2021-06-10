import torch
import numpy as np


class AverageMeter(object):
    """ Computes and stores the average and current value. """

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

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    return res


def acc_top1(preds, labels):
    acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def acc_cifar(preds, labels):
    overall_acc = acc_top1(preds, labels)
    cls_acc = []
    for i in range(10):
        cls_acc.append(acc_top1(preds[labels == i], labels[labels == i]))
    return overall_acc, cls_acc


def get_cls_count(train_data):
    training_labels = torch.tensor(train_data.targets).long()
    length = training_labels.max()+1
    cls_count = torch.zeros(length).long()
    for i in range(len(training_labels)):
        cls_count[training_labels[i]] += 1
    return cls_count


def shot_acc(preds, labels, train_class_count, many_shot_thr=100, low_shot_thr=10):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def mic_acc_cal(preds, labels):
    acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1
