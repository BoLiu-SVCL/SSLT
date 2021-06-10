from .metrics import AverageMeter, accuracy, acc_top1, get_cls_count, acc_cifar, shot_acc, mic_acc_cal
from .io_models import save_checkpoint


def adjust_learning_rate(optimizer, epoch, base_lr):
    epoch = epoch + 1
    if epoch <= 5:
        lr = base_lr * epoch / 5
    elif epoch > 160:
        lr = base_lr * 0.01
    elif epoch > 180:
        lr = base_lr * 0.0001
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

