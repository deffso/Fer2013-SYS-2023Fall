import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from io import BytesIO
import multiprocessing
from dataset import get_dataloaders
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os

from torchviz import make_dot


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1).to(device)
        self.gamma = nn.Parameter(torch.zeros(1)).to(device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width)
        proj_query = proj_query.permute(0, 2, 1)
        
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x

        return out


class ModifiedResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(ModifiedResNet, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet18.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1),SelfAttention(512))
        fc_in_features = resnet18.fc.in_features
        resnet18.fc = nn.Linear(fc_in_features, num_classes)
        self.to(device)
        
        self.features = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            SelfAttention(256),  # Adjust based on the output channels of your last layer
            resnet18.layer4,
           
            resnet18.avgpool
        )
        self.fc = resnet18.fc

    def forward(self, x):
        #x = self.features(x)
        x = x.to(self.features[0].weight.dtype)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

image_path = "D:\模式识别\Fer2013-Facial-Emotion-Recognition-Pytorch-main\ParrtenRegination\image_comparison_original\original_5.png"
image = Image.open(image_path)
# 创建 SelfAttention 层的虚构张量并进行图形可视化
# 创建 SelfAttention 层的虚构张量并进行图形可视化
dummy_tensor = torch.zeros((1, 512, 7, 7)).to(device)  # 移动到与模型相同的设备上
dummy_attention = SelfAttention(512)
dummy_tensor = dummy_tensor.to(device)  # 将 dummy_tensor 移动到 GPU 上
dummy_output = dummy_attention(dummy_tensor)
dot_attention = make_dot(dummy_output, params=dict(dummy_attention.named_parameters()))
dot_attention.render("self_attention", format="pdf", cleanup=True)
resnet18 = ModifiedResNet(in_channels=1, num_classes=7)
resnet18 = resnet18.to(device)
# 定义输入
example_input = torch.randn(1, 1, 48, 48).to(device)

# 获取模型输出
model_output = resnet18(example_input)

# 使用torchviz创建图
# 使用torchviz创建图
dot = make_dot(model_output, params=dict(resnet18.named_parameters()))

# 指定Graphviz可执行文件路径
dot.render("resnet18_attention", format="pdf", cleanup=True, executable='C:/Program Files/Graphviz/bin/dot.exe')


#dot = make_dot(model_output, params=dict(resnet18.named_parameters()))

# 保存图为PDF或其他格式
#dot.render("resnet18_attention", format="pdf", cleanup=True)
import hiddenlayer as hl

# 使用hiddenlayer创建注意力层图
hl_graph = hl.build_graph(resnet18, torch.zeros(example_input.shape).to(device))
hl_graph.save("resnet18_attention.png", format="png")

