import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class GenericDataset(data.Dataset):
    def __init__(self, root, dataset_name, split):
        self.root = root
        self.split = [s.lower() for s in split]
        self.dataset_name = dataset_name.lower()

        if self.dataset_name == 'imagenet':
            self.n_classes = 1000
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            transform = []
            if self.split[0] != 'test':
                transform.append(transforms.RandomResizedCrop(224))
                transform.append(transforms.RandomHorizontalFlip())
            else:
                transform.append(transforms.Resize(256))
                transform.append(transforms.CenterCrop(224))
            transform.append(transforms.ToTensor())
            transform.append(transforms.Normalize(self.mean_pix, self.std_pix))
            self.transform = transforms.Compose(transform)

            self.img_path = []
            self.targets = []
            self.num_per_cls_list = [0] * self.n_classes
            self.num_labels = 0
            for i in range(len(self.split)):
                if i > 0 and self.num_labels == 0:
                    self.num_labels = len(self.targets)
                with open('./data/ImageNet_SSLT/ImageNet_SSLT_'+split[i]+'.txt') as f:
                    for line in f:
                        self.img_path.append(line.split()[0])
                        self.targets.append(int(line.split()[1]))
                        self.num_per_cls_list[int(line.split()[1])] += 1

        elif self.dataset_name == 'inat':
            self.n_classes = 8080
            self.mean_pix = [0.466, 0.470, 0.380]
            self.std_pix = [0.195, 0.194, 0.192]
            transform = []
            if self.split[0] != 'test':
                transform.append(transforms.RandomResizedCrop(224))
                transform.append(transforms.RandomHorizontalFlip())
            else:
                transform.append(transforms.Resize(256))
                transform.append(transforms.CenterCrop(224))
            transform.append(transforms.ToTensor())
            transform.append(transforms.Normalize(self.mean_pix, self.std_pix))
            self.transform = transforms.Compose(transform)

            self.img_path = []
            self.targets = []
            self.num_per_cls_list = [0] * self.n_classes
            self.num_labels = 0
            for i in range(len(self.split)):
                if i > 0 and self.num_labels == 0:
                    self.num_labels = len(self.targets)
                with open('./data/iNaturalist_SSLT/iNaturalist_SSLT_'+split[i]+'.txt') as f:
                    for line in f:
                        self.img_path.append(line.split()[0])
                        self.targets.append(int(line.split()[1]))
                        self.num_per_cls_list[int(line.split()[1])] += 1

        else:
            raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))

    def update_pseudo_labels(self, pseudo_labels):
        if self.split[0] == 'train':
            self.targets = self.targets[:self.num_labels] + pseudo_labels

        elif self.split[0] == 'ext':
            self.targets = pseudo_labels

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]
        with open(os.path.join(self.root, path), 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.targets)

