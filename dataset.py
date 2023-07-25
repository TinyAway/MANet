from torch.utils.data import Dataset
import torch
import PIL.Image as Image
import os
import numpy as np
import pickle
import albumentations as albu
from torch.utils.data import DataLoader


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        print(self.class_ids)
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks = self.load_frame(query_name, support_names)
        # query_img = self.transform(query_img)
        # if not self.use_original_imgsize:
        #     query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)
        support_imgs = torch.stack([support_img for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask.float(), class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img, query_mask = self.read_imgpair(query_name, self.use_original_imgsize)
        support_imgs = []
        support_masks = []
        for name in support_names:
            support_img, support_mask = self.read_imgpair(name)
            support_imgs.append(support_img)
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks

    def read_imgpair(self, img_name, use_ori_imsize=False):
        """load images and masks after augmentation"""
        image = np.array(Image.open(os.path.join(self.img_path, img_name) + '.jpg'))
        mask = np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png'))
        augmented = self.transform(image=image, mask=mask)
        image = torch.tensor(np.transpose(augmented["image"], [2, 0, 1]))
        if use_ori_imsize:
            mask = torch.tensor(mask)
        else:
            mask = torch.tensor(augmented["mask"])
        return image, mask

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = ('G:/python/hsnet-main/data/splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = datapath
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 20000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample = self.load_frame()
        support_imgs = torch.stack(support_imgs)
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open('G:/python/hsnet-main/data/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_imgpair(self, img_name, use_ori_imsize=False):
        """load images and masks after augmentation"""
        image = np.array(Image.open(os.path.join(self.base_path, img_name)).convert('RGB'))
        mask = self.read_mask(img_name)
        augmented = self.transform(image=image, mask=mask)
        image = torch.tensor(np.transpose(augmented["image"], [2, 0, 1]))
        if use_ori_imsize:
            mask = torch.tensor(mask)
        else:
            mask = torch.tensor(augmented["mask"])
        return image, mask.float()

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png'))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img, query_mask = self.read_imgpair(query_name, self.use_original_imgsize)
        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_img, support_mask = self.read_imgpair(support_name, False)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_imgs.append(support_img)
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform_trn = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(
                scale_limit=0.1,
                rotate_limit=10,
                shift_limit=0.1,
                p=0.5,
                border_mode=0
            ),
            albu.GridDistortion(p=0.5),
            albu.Resize(img_size, img_size),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        cls.transform_val = albu.Compose([
            albu.Resize(img_size, img_size),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        if split == 'trn':
            shuffle = True
            nworker = nworker
            transform = cls.transform_trn
        else:
            shuffle = True
            nworker = 0
            transform = cls.transform_val

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split, shot=shot,
                                          use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader


if __name__ == "__main__":
    FSSDataset.initialize(img_size=473, datapath="F:/datasets/VOCtrainval_11-May-2012/VOCdevkit/pascal-5-master",
                          use_original_imgsize=True)
    dataloader_val = FSSDataset.build_dataloader('pascal', 1, 0, 1, 'val', 1)
