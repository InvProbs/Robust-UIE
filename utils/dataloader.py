from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch, glob
import numpy as np
from PIL import Image


class CustomDataset_UIEB(Dataset):
    def __init__(self, scene_path, gt_path, isPatched, transform=None):
        self.scene_img_path = scene_path
        self.gt_path = gt_path
        self.transform = transform
        self.isPatched = isPatched

        scene_file_list = glob.glob(scene_path + "/*")
        self.scene_data = []
        for img_path in scene_file_list:
            self.scene_data.append(img_path)
        self.scene_data.sort()

        gt_file_list = glob.glob(gt_path + "/*")
        self.gt_data = []
        for img_path in gt_file_list:
            self.gt_data.append(img_path)
        self.gt_data.sort()

    def __len__(self):
        return len(self.scene_data) * 16 if self.isPatched else len(self.scene_data)

    def __getitem__(self, idx):
        if self.isPatched:
            n_pathces = (4, 4)  # in row, column directions, 16 patches/image
            img_idx, patch_idx = idx // np.prod(n_pathces), np.mod(idx, np.prod(n_pathces))  # index for multiscene images
            row, col = patch_idx // n_pathces[1], np.mod(patch_idx, n_pathces[1])
            scene_img_path_current, gt_img_path_current = self.scene_data[img_idx], self.gt_data[img_idx]
            n1, n2 = 270 // n_pathces[0], 384 // n_pathces[1]
            scene_img = self.transform(Image.open(scene_img_path_current))[:3, row * n1:(row + 1) * n1, col * n2:(col + 1) * n2]
            gt_img = self.transform(Image.open(gt_img_path_current))[:3, row * n1:(row + 1) * n1, col * n2:(col + 1) * n2]
            return scene_img, gt_img
        else:
            scene_img_path_current, gt_img_path_current = self.scene_data[idx], self.gt_data[idx]
            scene_img = self.transform(Image.open(scene_img_path_current))[:3]
            gt_img = self.transform(Image.open(gt_img_path_current))[:3]
            return scene_img, gt_img


class SingleImageDataset(Dataset):
    def __init__(self, raw_path, gt_path, transform=None):
        self.raw_path = raw_path
        self.gt_path = gt_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        raw = Image.open(self.raw_path).convert("RGB")
        gt = Image.open(self.gt_path).convert("RGB")
        if self.transform:
            raw = self.transform(raw)
            gt = self.transform(gt)
        return raw, gt, raw


def create_single_image_dataloader(raw_path, gt_path, batch_size=1):
    transform = transforms.Compose([transforms.Resize((270, 384)),transforms.ToTensor(),])

    dataset = SingleImageDataset(raw_path, gt_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def load_UIEB(args):
    scene_path = '../data/UIEB/raw-890'
    gt_path = '../data/UIEB/reference-890'

    transform = transforms.Compose([transforms.Resize((270, 384)), transforms.ToTensor()])
    dataset = CustomDataset_UIEB(scene_path, gt_path, isPatched=False, transform=transform)

    full_length = len(dataset)
    tr_length, val_length, test_length = int(full_length * 0.8), int(full_length * 0.1), int(full_length * 0.1)
    train_index = range(int(tr_length * args.data_portion))
    val_index = range(tr_length, tr_length + val_length)
    test_index = range(tr_length + val_length, full_length)

    train_set = torch.utils.data.Subset(dataset, train_index)
    val_set = torch.utils.data.Subset(dataset, val_index)
    test_set = torch.utils.data.Subset(dataset, test_index)

    tr_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size_val, shuffle=False, drop_last=True)
    ts_loader = DataLoader(dataset=test_set, batch_size=args.batch_size_val, shuffle=False, drop_last=False)

    return tr_loader, len(train_set), val_loader, len(val_set), ts_loader, len(test_set)
