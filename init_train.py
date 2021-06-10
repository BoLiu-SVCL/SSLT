import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from tqdm import tqdm

import models
import dataset
from utils import get_cls_count, mic_acc_cal, shot_acc

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

prefix = 'CB_resnet18_imagenet'
weight_dir = os.path.join('weights', prefix)
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)

batch_size = 512
n_epoch = 200
n_class = 1000
n_feat = 512

data_root = '/your/ImageNet/path'
train_set = dataset.GenericDataset(root=data_root, dataset_name='imagenet', split=['train', 'ext'])
meta_set = dataset.GenericDataset(root=data_root, dataset_name='imagenet', split=['train'])
ext_set = dataset.GenericDataset(root=data_root, dataset_name='imagenet', split=['ext'])
test_set = dataset.GenericDataset(root=data_root, dataset_name='imagenet', split=['test'])


num_per_cls_list = get_cls_count(meta_set)
sampler_weights = [0.0] * len(meta_set)
for i in range(len(meta_set)):
    sampler_weights[i] = 1. / num_per_cls_list[meta_set.targets[i]].item()
meta_sampler = torch.utils.data.WeightedRandomSampler(weights=sampler_weights, num_samples=len(meta_set),
                                                      replacement=True)
randomloader = torch.utils.data.DataLoader(meta_set, batch_size=batch_size, shuffle=True, num_workers=16)
metaloader = torch.utils.data.DataLoader(meta_set, batch_size=batch_size, num_workers=16, sampler=meta_sampler)
extloader = torch.utils.data.DataLoader(ext_set, batch_size=batch_size, shuffle=False, num_workers=16)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=16)

cls_count = get_cls_count(meta_set)

device = 'cuda:0'

model, cls_model = models.load_net('ResNet-18-block', n_class)
model = model.to(device)
model = nn.DataParallel(model)
cls_model.to(device)
cls_model = nn.DataParallel(cls_model)

ce_loss = nn.CrossEntropyLoss()

optim_params_list = [{'params': model.parameters(), 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005},
                     {'params': cls_model.parameters(), 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005}]
optimizer = optim.SGD(optim_params_list)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=0.0)

print('Start training 1:')
for epoch in range(n_epoch):
    model.train()
    cls_model.train()
    loss_all = 0.0
    for i, (images, labels, path) in enumerate(tqdm(randomloader())):
        images, labels = images.to(device), labels.to(device)

        feat_map = model(images)
        logits = cls_model(feat_map)
        loss = ce_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += loss.item()

    scheduler.step()
    loss_all /= len(metaloader())
    print('Epoch[{}, {}] loss: {:.4f}'.format(epoch, n_epoch, loss_all))

cls_epoch = 20
_, cls_model = models.load_net('ResNet-18-block', n_class)
cls_model.to(device)
cls_model = nn.DataParallel(cls_model)
optim_params_list = [{'params': cls_model.parameters(), 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005}]
optimizer = optim.SGD(optim_params_list)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cls_epoch, eta_min=0.0)
print('Start training 2:')
for epoch in range(cls_epoch):
    model.eval()
    cls_model.train()
    loss_all = 0.0
    for i, (images, labels, path) in enumerate(tqdm(metaloader())):
        images, labels = images.to(device), labels.to(device)

        feat_map = model(images).detach()
        logits = cls_model(feat_map)
        loss = ce_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += loss.item()

    scheduler.step()
    loss_all /= len(metaloader())
    print('Epoch[{}, {}] loss: {:.4f}'.format(epoch, cls_epoch, loss_all))

    if (epoch % 5 == 0) or (epoch == cls_epoch - 1):
        torch.save({'feat': model.state_dict(), 'cls': cls_model.state_dict()}, '{}/{}_epoch_{}.pt'.format(weight_dir, prefix, epoch))

        model.eval()
        cls_model.eval()

        total_logits = torch.empty((0, n_class)).to(device)
        total_labels = torch.empty(0, dtype=torch.long).to(device)

        with torch.no_grad():
            for i, (images, labels, path) in enumerate(tqdm(testloader())):
                images, labels = images.to(device), labels.to(device)
                feat_map = model(images)
                logits = cls_model(feat_map)

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, labels))

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)

        eval_acc = mic_acc_cal(preds[total_labels != -1], total_labels[total_labels != -1])
        many_acc, median_acc, low_acc = shot_acc(preds[total_labels != -1], total_labels[total_labels != -1], cls_count)
        # Top-1 accuracy and additional string
        print_str = 'Init: Epoch[{}, {}] Evaluation_accuracy: {:.3f}\n' \
                    'Many_shot_accuracy: {:.3f} Median_shot_accuracy: {:.3f} ' \
                    'Low_shot_accuracy: {:.3f}'.format(epoch, cls_epoch, eval_acc, many_acc, median_acc, low_acc)
        print(print_str)

        # save results into a txt file
        text_file = open(os.path.join(weight_dir, 'results.txt'), 'a')
        text_file.write(print_str)
        text_file.write('\n')
        text_file.close()
