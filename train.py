# encoding=utf-8
import argparse
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MyDataset
from net import EmotionNet
from test import get_f1_score
from training_utils import EarlyStopping


class Processor(object):
    """
    调度器
    """

    def __init__(self, log_dir="log"):
        """
        构造函数，初始化tensorboard
        """
        self.trainWriter = SummaryWriter(f"{log_dir}/emotion")
        self.global_step = 0
        self.softmax = nn.Softmax(dim=1)

    def train(self, model, optimizer, data_loader, device, epoch):
        """

        :param device:
        :param data_loader:
        :param optimizer:
        :param model:
        :type epoch: int
        """
        criterion = nn.CrossEntropyLoss()
        model.train()

        sum_loss = 0
        for batch_idx, (X, y) in enumerate(data_loader, 0):
            # 获取数据并前向传播
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()

            optimizer.step()
            sum_loss += loss.item()
            self.global_step += 1
            self.trainWriter.add_scalar("loss", loss.item(), self.global_step)
        self.trainWriter.add_scalar("train loss", sum_loss / len(data_loader), epoch + 1)
        # 获取当前学习率并记录
        lr = optimizer.param_groups[0]['lr']
        self.trainWriter.add_scalar("lr", lr, epoch + 1)

    def valid(self, model, device, valid_loader, epoch):
        list_output = []
        list_y = []
        model.eval()
        valid_loss_sum = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for X, y in valid_loader:
                torch.cuda.empty_cache()
                X = X.to(device)
                y = y.to(device)
                output = model(X)
                loss = criterion(output, y)
                valid_loss_sum += loss.item()
                output = self.softmax(output)
                output = torch.argmax(output, dim=1)
                list_output.extend(output.cpu().tolist())
                list_y.extend(y.cpu().tolist())

                # print(list_y, list_output)
        valid_loss = valid_loss_sum / len(valid_loader)
        print('\nvalid set: Average loss: {:.4f},({:.0f})\n'.format(valid_loss, len(valid_loader.dataset)))
        self.trainWriter.add_scalar("valid loss", valid_loss, epoch + 1)

        f1 = get_f1_score(output=list_output, y=list_y)

        self.trainWriter.add_scalar("valid f1-score", f1, epoch + 1)

        return valid_loss

    def start(self, lr, batch_size, max_epochs, load_param, path_train_X, path_valid_X):
        """
        启动训练任务
        :param lr: 学习率
        :param batch_size:批次大小
        :param max_epochs: 最大轮次
        :param load_param: 是否加载预训练参数
        :param path_train_X: 训练集路径
        :param path_valid_X: 验证集路径
        :return:
        """

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        kwargs = {'pin_memory': True, "num_workers": 1} if use_cuda else {}  # 锁页内存

        train_transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomHorizontalFlip(p=0.5), # 不要垂直翻转，严重影响收敛速度
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5),
            transforms.ToTensor(),
            # to_float32,
            transforms.Normalize([0.5, ], [0.24, ])

        ])
        valid_transforms = transforms.Compose([
            transforms.ToTensor(),
            # to_float32,
            transforms.Normalize([0.5, ], [0.24, ])

        ])

        num_classes = 7
        net = EmotionNet(num_classes=num_classes)
        # net = EmoNeXt(num_classes=num_classes)

        if load_param:
            PATH = "./models/best_network.pth"
            state_dict = torch.load(PATH, map_location=device)
            net.load_state_dict(state_dict)
            print("参数初始化完成")
        else:
            # 设置随机种子
            seed = random.randint(0, 2 ** 32 - 1)
            print("随机种子:", seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 在使用GPU时，还需要设置以下语句
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # 确保每次运行结果一致
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print("加载数据集…………")
        train_dataset = MyDataset(path_X=path_train_X, transform=train_transform)
        valid_dataset = MyDataset(path_X=path_valid_X, transform=valid_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=True, drop_last=False, **kwargs)
        print("数据加载完成")
        net = net.to(device)
        optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
        # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # 每step_size个epoch lr变为原来的gamma倍
        save_path = "./models"
        early_stopping = EarlyStopping(save_path=save_path, patience=10)

        for epoch in range(max_epochs):
            print(epoch)
            self.train(model=net, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch)
            scheduler.step()
            if epoch % 1 == 0:
                valid_loss = self.valid(model=net, valid_loader=valid_loader, device=device, epoch=epoch)
                early_stopping(valid_loss, net)
                # 达到早停止条件时，early_stop会被置为True
                if early_stopping.early_stop:
                    print("Early stopping")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
    parser.add_argument('--max_epochs', type=int, default=120, help='max_epochs')
    parser.add_argument('--load_param', action="store_true", help='if load last params')
    parser.add_argument('--path_train_X', type=str, default="data/FER-2013/train", help='train set')
    parser.add_argument('--path_valid_X', type=str, default="data/FER-2013/test", help='valid set')
    args = parser.parse_args()
    print(args)
    lr_ = args.lr
    batch_size_ = args.batch_size
    max_epochs_ = args.max_epochs
    load_param_ = args.load_param
    path_train_X_ = args.path_train_X
    path_valid_X_ = args.path_valid_X
    processor = Processor(log_dir=f"./log/{datetime.date.today()}/{datetime.datetime.now().timestamp()}"[:-7])
    processor.start(lr=lr_, batch_size=batch_size_, max_epochs=max_epochs_, load_param=load_param_,
                    path_train_X=path_train_X_, path_valid_X=path_valid_X_)
