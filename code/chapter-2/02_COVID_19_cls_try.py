# -*- coding: utf-8 -*-

"""
@author: howhaokw
@date: 2023-06-17
@brief: python tutorial classify COVID-19
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

def main():
    """
    step 1: data
    step 2: model
    step 3: optimizer
    step 4: train
    """

    # step 1: data
    class COVID19Dataset(Dataset):
        def __init__(self, root_dir, txt_path, transform=None):
            self.root_dir = root_dir
            self.txt_path = txt_path
            self.transform = transform
            self.img_info = [] # [(path, label), ...]
            self.lable_array = None

            self._get_img_info()

        def __getitem__(self, index):
            path_img, label = self.img_info[index]
            img = Image.open(path_img).convert('L')

            if self.transform is not None:
                img = self.transform(img)

            return img, label

        def __len__(self):
            if len(self.img_info) == 0:
                raise Exception(f'data_dir: {self.root_dir} is a empty dir!'
                                f'Please checkout your path to images!')
            return len(self.img_info)

        def _get_img_info(self):
            with open(self.txt_path, 'r') as f:
                txt_data = f.read().strip()
                txt_data = txt_data.split("\n")

            self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
                             for i in txt_data]

    file_dir = os.path.dirname(__file__)
    root_dir = os.path.join(file_dir, r'../../data/datasets/covid-19-demo')
    img_dir = os.path.join(root_dir, 'imgs')
    path_txt_train = os.path.join(root_dir, 'labels', 'train.txt')
    path_txt_valid = os.path.join(root_dir, 'labels', 'valid.txt')
    transforms_func = transforms.Compose([
        transforms.Resize((8, 8)),
        transforms.ToTensor(),
    ])

    train_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_func)
    valid_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_valid, transform=transforms_func)
    train_loader = DataLoader(dataset=train_data, batch_size=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=2)

    # setp 2: model
    class TinnyCNN(nn.Module):
        def __init__(self, cls_num=2):
            super(TinnyCNN, self).__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=(3, 3))
            # 8 - 3 + 1 = 6
            self.fc = nn.Linear(36, cls_num)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            out = self.fc(x)
            return out

    model = TinnyCNN(2)

    # setp 3: optimizer
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # step 4: train
    for epoch in range(100):
        model.train()
        for data, labels in train_loader:
            outputs = model(data)
            optimizer.zero_grad()

            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct_num = (predicted == labels).sum()
            acc = correct_num / labels.shape[0]
            print(f'Epoch: {epoch}, Train Loss: {loss:.2f}, Acc: {acc:.0%}')
            print(predicted, labels)

        # valid
        for data, label in valid_loader:
            outputs = model(data)

            loss = loss_f(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            correct_num = (predicted == labels).sum()
            acc_valid = correct_num / labels.shape[0]
            print(f'Epoch: {epoch}, Valid Loss: {loss:.2f}, Acc: {acc_valid:.0%}')

        if acc_valid == 1:
            break

        scheduler.step()


if __name__ == '__main__':
    main()
