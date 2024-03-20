# encoding=utf-8
import os

import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from net import FACE_SHAPE


class MyDataset(Dataset):
    def __init__(self, path_X, path_y=None, transform=None, target_transform=None):
        """
        自定义数据集
        Args:
            path_X: 特征文件路径
            path_y:标签文件路径
            transform: X的归一化规则
            target_transform: y的转换规则
        """
        self.path_X = path_X
        self.path_y = path_y
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.labels = []
        class_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprise": 6,
        }
        for dir_name in tqdm(os.listdir(path_X)):
            for filename in os.listdir(f"{path_X}/{dir_name}"):
                self.labels.append(class_map[dir_name])
                image = Image.open(f"{path_X}/{dir_name}/{filename}").convert("L")
                image = image.resize(FACE_SHAPE)
                self.images.append(image)
        # print(self.labels)

    def __getitem__(self, index):
        """
        读1条样本
        Args:
            index: 样本下标

        Returns: X，y

        """
        X = self.images[index]
        y = self.labels[index]

        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return X, y

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # 测试自定义数据集
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.24, ])
    ])
    dataset = MyDataset(path_X="./data/FER-2013/test", transform=trans)
    print(dataset.__len__())
