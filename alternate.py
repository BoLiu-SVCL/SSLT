import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import numpy as np
from tqdm import tqdm

import models
import dataset
from utils import get_cls_count, mic_acc_cal, shot_acc


def train(args, model, cls, train_loader, optimizer, epoch, meta_or_train, criterion):
    device = args.device

    if (args.loop > 0) and (meta_or_train == 'meta'):
        model.eval()
        cls.train()
    else:
        model.train()
        cls.train()

    # Training loop, train with cross entropy and consistency losses
    sum_loss = 0
    correct, total_seen = 0, 0
    for batch_idx, data_batch in enumerate(train_loader()):
        if meta_or_train == 'train':
            data, target, index_batch = data_batch

            meta_labels = args.meta_labels_total[index_batch].clone()
            meta_labels.requires_grad = False
            data, target, meta_labels = data.to(device), target.to(device), meta_labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = cls(output)
            loss = -torch.mean(torch.sum(meta_labels * F.log_softmax(output, dim=1), 1)) + criterion(output, target)

        elif meta_or_train == 'meta':
            data, target, index_batch = data_batch
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data).detach()
            output = cls(output)
            loss = criterion(output, target)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_seen += len(target)
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
        if meta_or_train == 'train':
            args.meta_labels_total[index_batch] = F.softmax(output, dim=1).data.cpu()  # Store the predictions for the next epoch for self distillation
    if meta_or_train == 'train':
        print('Train Epoch: {}\tLoss: {:.6f}'.format(
            epoch, sum_loss / len(train_loader)
        ))


def fine_tune_and_assign_labels(args, model, cls, metaloader, testloader, extloader, train_set, criterion):
    device = args.device

    if args.loop == 0:
        print('No finetuning in loop 0')
    else:
        print('Finetune the classifier layer with labeled samples')
        optimizer = optim.SGD(cls.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        nb_epochs = args.epochs_cls

        # Training loop
        for epoch in tqdm(range(nb_epochs)):
            train(args, model, cls, metaloader, optimizer, epoch, 'meta', criterion)
    test(args, model, cls, testloader)

    # Assign pseudo labels by the model
    model.eval()
    cls.eval()
    size_ext = len(extloader.dataset)
    pseudo_targets = torch.zeros(size_ext).type(torch.LongTensor)
    correct_total = 0
    with torch.no_grad():
        for batch_idx, (data, target, index_batch) in enumerate(extloader()):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = cls(output)
            _, pred = output.max(dim=1)
            correct_total += pred.eq(target).sum().item()
            pseudo_targets[index_batch] = pred.detach().cpu()

    train_set.update_pseudo_labels(pseudo_targets.flatten().tolist())

    print('Correctly labelled data on ext set {}'.format(100. / size_ext * correct_total))


def final_test(args, model, cls, metaloader, testloader, criterion):

    print('Final classifier training')
    optimizer = optim.SGD(cls.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    nb_epochs = args.epochs_cls
    # Training loop
    for epoch in tqdm(range(nb_epochs)):
        train(args, model, cls, metaloader, optimizer, epoch, 'meta', criterion)

    test(args, model, cls, testloader)


def test(args, model, cls, test_loader):
    device = args.device

    # Compute test loss
    model.eval()
    cls.eval()
    test_loss = 0
    size_test = len(test_loader.dataset)
    total_logits = torch.zeros((size_test, test_loader.dataset.n_classes))
    total_labels = torch.zeros(size_test, dtype=torch.long)
    with torch.no_grad():
        for (data, target, index_batch) in test_loader():
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = cls(output)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()  # sum up batch loss

            total_logits[index_batch, :] = output.detach().cpu()
            total_labels[index_batch] = target.detach().cpu()

    args.test_loss.append(test_loss / len(test_loader.dataset))
    print('Test set: Average loss: {:.4f}'.format(
        test_loss / len(test_loader.dataset)
    ))

    _, preds = total_logits.max(dim=1)

    overall_acc = mic_acc_cal(preds[total_labels != -1], total_labels[total_labels != -1])
    many_acc, median_acc, low_acc = shot_acc(preds[total_labels != -1], total_labels[total_labels != -1], args.cls_count)
    # Top-1 accuracy
    print_str = 'Loop {}: Evaluation_accuracy: {:.3f}\n' \
                'Many_shot_accuracy: {:.3f} Median_shot_accuracy: {:.3f} ' \
                'Low_shot_accuracy: {:.3f}'.format(args.loop, overall_acc, many_acc, median_acc, low_acc)
    print(print_str)

    if overall_acc > args.best_acc:
        args.best_acc = overall_acc
        torch.save({'feat': model.state_dict(), 'cls': cls.state_dict()},
                   args.best_path)
        text_file = open(os.path.join(args.save_dir, 'best_results.txt'), 'a')
        text_file.write(print_str)
        text_file.write('\n')
        text_file.close()

    # save results into a txt file
    text_file = open(os.path.join(args.save_dir, 'results.txt'), 'a')
    text_file.write(print_str)
    text_file.write('\n')
    text_file.close()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Alternative Training for Semi-supervised learning')
    parser.add_argument('--batch_size', type=int, default=384, metavar='N',
                        help='Input batch size for training (default: 384)')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='Dataset (default: imagenet)')
    parser.add_argument('--epochs_cls', type=int, default=20,
                        help='number of random epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default 0.01)')
    parser.add_argument('--nloops', nargs='+', type=int, default=5,
                        help='number of loops')
    parser.add_argument('--CB_nepochs', nargs='+', type=int, default=[40, 50],
                        help='number of class-balanced epochs (change of lr and number of epochs)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--network', type=str, default='ResNet-18',
                        help='Network (default: ResNet-18)')
    parser.add_argument('--init_dir', type=str, default='',
                        help='Initialization directory')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default='./weight/alternate/',
                        help='Directory to save models')
    parser.add_argument('--root', type=str, default='/your/ImageNet/path',
                        help='dataset root')
    parser.add_argument('--data_parallel', action='store_true', default=True,
                        help='Multi GPU')
    args = parser.parse_args()

    args.init_dir = './weights/CB_resnet18_imagenet'
    args.model_name = 'CB_resnet18_imagenet_epoch_19.pt'
    args.batch_size = 512
    args.save_dir = './weights/alternative_CB_resnet18_imagenet'
    args.network = 'ResNet-18'

    # Path to file
    os.makedirs(args.save_dir, exist_ok=True)
    args.name = 'alternate'
    args.net_path = os.path.join(args.save_dir, args.name + '.pth')
    args.best_path = os.path.join(args.save_dir, args.name + '_best.pth')

    args.test_loss = []

    # Set up seed and GPU usage
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device

    # Initialize the dataset
    train_set = dataset.GenericDataset(root=args.root, dataset_name=args.dataset, split=['train', 'ext'])
    meta_set = dataset.GenericDataset(root=args.root, dataset_name=args.dataset, split=['train'])
    ext_set = dataset.GenericDataset(root=args.root, dataset_name=args.dataset, split=['ext'])
    test_set = dataset.GenericDataset(root=args.root, dataset_name=args.dataset, split=['test'])

    args.cls_count = get_cls_count(meta_set)

    criterion = nn.CrossEntropyLoss()

    cls_count = get_cls_count(meta_set)
    sampler_weights = [0.0] * len(meta_set.targets)
    for i in range(len(meta_set.targets)):
        sampler_weights[i] = 1. / cls_count[meta_set.targets[i]].item()
    meta_sampler = torch.utils.data.WeightedRandomSampler(weights=sampler_weights, num_samples=len(meta_set.targets), replacement=True)
    metaloader = torch.utils.data.DataLoader(meta_set, batch_size=args.batch_size, num_workers=16, sampler=meta_sampler)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    extloader = torch.utils.data.DataLoader(ext_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=16)

    # Network Initialization
    model, cls = models.load_net(args.network+'-block', train_set.n_classes)
    model = model.to(device)
    cls = cls.to(device)
    if args.data_parallel:
        model = nn.DataParallel(model)
        cls = nn.DataParallel(cls)

    # Load model
    if args.init_dir:
        state_dict_init = torch.load(os.path.join(args.init_dir, args.model_name))
        model.load_state_dict(state_dict_init['feat'])
        cls.load_state_dict(state_dict_init['cls'])

    args.best_acc = 0.0
    for args.loop in range(0, args.nloops):
        print('Entering loop {}/{}'.format(args.loop, args.nloops))

        # Stage 1&3: fine-tune and assign labels
        fine_tune_and_assign_labels(args, model, cls, metaloader, testloader, extloader, train_set, criterion)

        args.meta_labels_total = torch.ones(len(trainloader.dataset), train_set.n_classes) / float(train_set.n_classes)

        # Optimizer and LR scheduler
        optimizer = optim.SGD(list(model.parameters())+list(cls.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.CB_nepochs[0]], gamma=0.1)

        # Stage 2: class-balanced fine-tune
        print('Labels assignment done. fine-tune feature embedding')
        for epoch in range(args.CB_nepochs[1]):
            train(args, model, cls, trainloader, optimizer, epoch, 'train', criterion)
            test(args, model, cls, testloader)
            scheduler.step()

        torch.save({'feat': model.state_dict(), 'cls': cls.state_dict()}, args.net_path)

    final_test(args, model, cls, metaloader, testloader, criterion)


if __name__ == '__main__':
    main()
