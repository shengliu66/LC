'''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''

import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms as T
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from .civilcomments import CivilComments, init_bert_transform
from transformers import BertTokenizerFast
class dataloader():  
    def __init__(self, dataset, data_dir, percent, data2preprocess, use_type0, use_type1, batch_size, num_workers):
        self.dataset = dataset
        self.data_dir = data_dir
        self.percent = percent
        self.data2preprocess = data2preprocess
        self.use_type0 = use_type0
        self.use_type1 = use_type1
        self.num_workers = num_workers
        self.batch_size = batch_size

        

    def run(self,mode,preds=[],probs=[]):
        if mode=='train':
            train_dataset = get_dataset(
                self.dataset,
                data_dir=self.data_dir,
                dataset_split="train",
                transform_split="train",
                percent=self.percent,
                use_preprocess=self.data2preprocess[self.dataset],
                use_type0=self.use_type0,
                use_type1=self.use_type1
            )

            Idx_train_dataset = IdxDataset(train_dataset)
               
            trainloader = DataLoader(
                Idx_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )          
            return trainloader, train_dataset
                                     
        
        elif mode=='valid':
            valid_dataset= get_dataset(
                self.dataset,
                data_dir=self.data_dir,
                dataset_split="valid",
                transform_split="valid",
                percent=self.percent,
                use_preprocess=self.data2preprocess[self.dataset],
                use_type0=self.use_type0,
                use_type1=self.use_type1
            )

            valid_dataset = IdxDataset(valid_dataset)

            valid_trainloader = DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )


            return valid_trainloader

        elif mode=='test':
            test_dataset= get_dataset(
                self.dataset,
                data_dir=self.data_dir,
                dataset_split="test",
                transform_split="valid",
                percent=self.percent,
                use_preprocess=self.data2preprocess[self.dataset],
                use_type0=self.use_type0,
                use_type1=self.use_type1
            )

            test_dataset = IdxDataset(test_dataset)

            test_trainloader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            return test_trainloader

        elif mode=='eval_train':
            eval_train_dataset = get_dataset(
                self.dataset,
                data_dir=self.data_dir,
                dataset_split="train",
                transform_split="train",
                percent=self.percent,
                use_preprocess=self.data2preprocess[self.dataset],
                use_type0=self.use_type0,
                use_type1=self.use_type1
            )

            eval_train_dataset = IdxDataset(eval_train_dataset)

            eval_train_trainloader = DataLoader(
                eval_train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False
            )
            return eval_train_trainloader
      


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

class CMNISTDataset(Dataset):
    def __init__(self,root,split,transform=None, image_path_list=None, preds = None, bias = True):
        super(CMNISTDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            data = self.align + self.conflict
            indicator = [0] * len(self.align) + [1] * len(self.conflict)

            # print(len(self.data),'***************')



            if (preds is not None):
                pred_idx = (preds).numpy().nonzero()[0]
                if bias:
                    print("Discovered biased example id", pred_idx)
                else:
                    print("Discovered unbiased example id", pred_idx)

                self.data = [data[i] for i in pred_idx] * int(len(data)/len(pred_idx)) 
                self.indicator = [indicator[i] for i in pred_idx] * int(len(data)/len(pred_idx)) 
            else:
                self.data = data 
                self.indicator = indicator

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*"))            
        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))
        self.split = split


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.split != 'train':
            return image, attr, self.data[index], -1
        else:
            return image, attr, self.data[index], self.indicator[index]


class CIFAR10Dataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None, use_type0=None, use_type1=None, preds = None, bias = True):
        super(CIFAR10Dataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            data = self.align + self.conflict
            indicator = [0] * len(self.align) + [1] * len(self.conflict)

            if (preds is not None):
                pred_idx = (preds).numpy().nonzero()[0]
                if bias:
                    print("Discovered biased example id", pred_idx)
                else:
                    print("Discovered unbiased example id", pred_idx)

                self.data = [data[i] for i in pred_idx] * int(len(data)/len(pred_idx)) 
                self.indicator = [indicator[i] for i in pred_idx] * int(len(data)/len(pred_idx)) 
            else:
                self.data = data 
                self.indicator = indicator

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))


        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')


        if self.transform is not None:
            image = self.transform(image)

        if self.split != 'train':
            return image, attr, self.data[index], -1
        else:
            return image, attr, self.data[index], self.indicator[index]


class bFFHQDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None, preds = None, bias = True):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root

        self.image2pseudo = {}
        self.image_path_list = image_path_list




        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            data = self.align + self.conflict
            indicator = [0] * len(self.align) + [1] * len(self.conflict)

            if (preds is not None):
                pred_idx = (preds).numpy().nonzero()[0]
                if bias:
                    print("Discovered biased example id", pred_idx)
                else:
                    print("Discovered unbiased example id", pred_idx)

                self.data = [data[i] for i in pred_idx] * int(len(data)/len(pred_idx)) 
                self.indicator = [indicator[i] for i in pred_idx] * int(len(data)/len(pred_idx)) 
            else:
                self.data = data 
                self.indicator = indicator


        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))

        elif split=='test':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict
        self.split = split
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.split != 'train':
            return image, attr, self.data[index], -1
        else:
            return image, attr, self.data[index], self.indicator[index]


class WaterBirdsDataset(Dataset): 
    def __init__(self, root, split="train", transform=None, image_path_list=None, preds = None, bias = True):
        try:
            split_i = ["train", "valid", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        self.split = split
        metadata_df = pd.read_csv(os.path.join(root, "metadata.csv"))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        self.root = root
        self.transform = transform
        self.y_array = self.metadata_df['y'].values
        self.p_array = self.metadata_df['place'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['place'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.indicator = np.abs(self.y_array  -  self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        attr = torch.LongTensor(
            [y, p, g])

        img_path = os.path.join(self.root, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        # img = img.float() / 255.



        if self.transform:
            img = self.transform(img)

        if self.split != 'train':
            return img, attr, self.filename_array[idx], self.indicator[idx]
        else:
            return img, attr, self.filename_array[idx], self.indicator[idx]
        



transforms = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224,224)), T.ToTensor()])
        },
    "cifar10c": {
        "train": T.Compose([T.ToTensor(),]),
        "valid": T.Compose([T.ToTensor(),]),
        "test": T.Compose([T.ToTensor(),]),
        },

    "waterbird":{
         "train": T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
         "valid": T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
         "test": T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
         },

    }


transforms_preprcs = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        },
    "cifar10c": {
        "train": T.Compose(
            [
                T.RandomCrop(32, padding=4),
                # T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
    "waterbird": {
        "train": T.Compose(
            [
                T.RandomResizedCrop(
                    (224,224),
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]    
        ),
        "valid": T.Compose(
            [
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
    },
} 



transforms_preprcs_ae = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    },
    "cifar10c": {
        "train": T.Compose(
            [
                # T.RandomCrop(32, padding=4),
                T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
    
    "waterbird":{
         "train": T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
         "valid": T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
         "test": T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
         },
}
def get_dataset(dataset, data_dir, dataset_split, transform_split, percent, use_preprocess=None, image_path_list=None, use_type0=None, use_type1=None, preds = None, bias = True):
    dataset_category = dataset.split("-")[0]
    if dataset != 'civilcomments':
        if use_preprocess:
            transform = transforms_preprcs[dataset_category][transform_split]
        else:
            transform = transforms[dataset_category][transform_split]
    else:
        arch = 'bert-base-uncased_pt'
        pretrained_name = arch if arch[-3:] != '_pt' else arch[:-3]
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_name)  # 'bert-base-uncased'
        transform = init_bert_transform(tokenizer, pretrained_name, max_token_length = 300)


    dataset_split = "valid" if (dataset_split == "eval") else dataset_split
    if dataset == 'cmnist':
        root = data_dir + f"/cmnist/{percent}"
        dataset = CMNISTDataset(root=root,split=dataset_split,transform=transform, image_path_list=image_path_list, preds = preds, bias = bias)

    elif 'cifar10c' in dataset:
        # if use_type0:
        #     root = data_dir + f"/cifar10c_0805_type0/{percent}"
        # elif use_type1:
        #     root = data_dir + f"/cifar10c_0805_type1/{percent}"
        # else:
        root = data_dir + f"/cifar10c/{percent}"
        dataset = CIFAR10Dataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list, use_type0=use_type0, use_type1=use_type1)

    elif dataset == "bffhq":
        root = data_dir + f"/bffhq/{percent}"
        dataset = bFFHQDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)
    elif dataset == 'waterbird':
        root = data_dir + f"/waterbird"
        dataset = WaterBirdsDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)
    elif dataset == 'civilcomments':
        root = data_dir + f"/CivilComments"
        if dataset_split == 'train':
            dataset = CivilComments(root, target_name='toxic',
                                  confounder_names=['identities'],
                                  split='train', transform=transform)
        elif dataset_split == 'valid':
            dataset = CivilComments(root, target_name='toxic',
                                confounder_names=['identities'],
                                split='val', transform=transform)
        elif dataset_split == 'test':
            dataset = CivilComments(root, target_name='toxic',
                                 confounder_names=['identities'],
                                 split='test', transform=transform)
    else:
        print('wrong dataset ...')
        import sys
        sys.exit(0)

    return dataset

