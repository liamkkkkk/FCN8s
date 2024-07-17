import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model_FCN8s import FCN8s  # 导入FCN8s类
from dataset import VOC2007Dataset  # 导入自定义的VOC2007数据集类
from torchvision import transforms
from PIL import Image

# 检查CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return transforms.functional.resize(image, self.size)

# 创建数据增强和标准化转换
image_transform = transforms.Compose([
    Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    Resize((256, 256)),
    transforms.ToTensor()
])

# 定义训练函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            running_loss = 0.0

            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()  # 清除梯度

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # 深拷贝最佳模型
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

    print('Best val Loss: {:4f}'.format(best_loss))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # 加载模型
    model = FCN8s(n_class=21)
    model.load_state_dict(torch.load('fcn8s_model.pth'))
    model = model.to(device)

    # 加载数据集对象
    voc_train = VOC2007Dataset(root_dir='./data/VOCdevkit/VOC2007', image_transform=image_transform, mask_transform=mask_transform)
    voc_val = VOC2007Dataset(root_dir='./data/VOCdevkit/VOC2007', image_set='val', image_transform=image_transform, mask_transform=mask_transform)

    # 创建DataLoader
    train_loader = DataLoader(voc_train, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(voc_val, batch_size=8, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

    # 保存训练好的模型
    torch.save(model.state_dict(), 'fcn8s_trained.pth')
