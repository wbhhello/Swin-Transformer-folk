import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据Swin Transformer的模型选择适当的输入尺寸
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CustomDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=data_transform):
        self.img_dir = img_dir
        self.labels_file = labels_file
        self.img_names = os.listdir(img_dir)  # 读取文件夹中的所有图片名称
        self.labels = [int(line.strip()) for line in open(labels_file, 'r').readlines()]  # 读取标签文件
        self.labels = torch.tensor(self.labels)  # 将标签转换为张量
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)  # 拼接图片路径
        # 读取JPEG图片
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        label = self.labels[idx]  # 获取标签
        if self.transform:
            img = self.transform(img)
        return img, label


# 假设你的数据集为custom_dataset，包含所有的图像和标签
def getData(img_dir, labels_file, transform):
    custom_dataset = CustomDataset(img_dir, labels_file, transform)
    total_size = len(custom_dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])
    return train_dataset, test_dataset


def getDataLoader(img_dir, labels_file, transform=None):
    train_dataset, test_dataset = getData(img_dir, labels_file, transform)
    batch_size = 16  # 设置适当的批量大小
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # for img, label in test_dataloader:
    #     print(label)
    print(len(test_dataloader))  # 313
    print(len(train_dataloader))  # 2813
    return train_dataloader, test_dataloader


# def main():
#     getDataLoader(transform=data_transform)


# if __name__ == main():
#     main()
