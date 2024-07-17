# import torch
# import torch.nn as nn
# from torchvision import models
# from torchvision.models import VGG16_Weights  # 引入VGG16权重
#
# # 定义FCN8s模型类
# class FCN8s(nn.Module):
#     def __init__(self, n_class=21):
#         super(FCN8s, self).__init__()
#         vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
#         self.features = vgg.features
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv1 = nn.Sequential(*vgg.features[:5])
#         self.conv2 = nn.Sequential(*vgg.features[5:10])
#         self.conv3 = nn.Sequential(*vgg.features[10:17])
#         self.conv4 = nn.Sequential(*vgg.features[17:24])
#         self.conv5 = nn.Sequential(*vgg.features[24:])
#
#         self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
#         self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
#         self.score_fr = nn.Conv2d(4096, n_class, kernel_size=1)
#
#         self.score_pool3 = nn.Conv2d(256, n_class, kernel_size=1)
#         self.score_pool4 = nn.Conv2d(512, n_class, kernel_size=1)
#
#         self.upscore2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, bias=False)
#         self.upscore8 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8, bias=False)
#         self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, bias=False)
#
#     def forward(self, x):
#         pool1 = self.conv1(x)
#         pool2 = self.conv2(pool1)
#         pool3 = self.conv3(pool2)
#         pool4 = self.conv4(pool3)
#         pool5 = self.conv5(pool4)
#
#         fc6 = self.fc6(pool5)
#         fc7 = self.fc7(self.relu(fc6))
#         score_fr = self.score_fr(self.relu(fc7))
#
#         score_pool4 = self.score_pool4(pool4)
#         upscore2 = self.upscore2(score_fr)
#         score_pool4 = score_pool4[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
#         upscore_pool4 = self.upscore_pool4(score_pool4 + upscore2)
#
#         score_pool3 = self.score_pool3(pool3)
#         score_pool3 = score_pool3[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
#         upscore8 = self.upscore8(score_pool3 + upscore_pool4)
#
#         return upscore8[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]]
#
# # 实例化模型并保存
# model = FCN8s(n_class=21)
# torch.save(model.state_dict(), 'fcn8s_model.pth')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights  # 引入VGG16权重

# 定义FCN8s模型类
class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(*vgg.features[:5])
        self.conv2 = nn.Sequential(*vgg.features[5:10])
        self.conv3 = nn.Sequential(*vgg.features[10:17])
        self.conv4 = nn.Sequential(*vgg.features[17:24])
        self.conv5 = nn.Sequential(*vgg.features[24:])

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.score_fr = nn.Conv2d(4096, n_class, kernel_size=1)

        self.score_pool3 = nn.Conv2d(256, n_class, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, n_class, kernel_size=1)

    def forward(self, x):
        pool1 = self.conv1(x)
        pool2 = self.conv2(pool1)
        pool3 = self.conv3(pool2)
        pool4 = self.conv4(pool3)
        pool5 = self.conv5(pool4)

        fc6 = self.fc6(pool5)
        fc7 = self.fc7(self.relu(fc6))
        score_fr = self.score_fr(self.relu(fc7))

        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore2 = F.interpolate(score_fr, size=score_pool4.shape[2:], mode='bilinear', align_corners=True)
        upscore_pool4 = F.interpolate(score_pool4 + upscore2, size=score_pool3.shape[2:], mode='bilinear', align_corners=True)
        upscore8 = F.interpolate(score_pool3 + upscore_pool4, size=x.shape[2:], mode='bilinear', align_corners=True)

        return upscore8

# 实例化模型并保存
model = FCN8s(n_class=21)
torch.save(model.state_dict(), 'fcn8s_model.pth')

