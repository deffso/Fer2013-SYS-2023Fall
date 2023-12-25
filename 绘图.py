import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
print(device)
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
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
        resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
        
        self.features = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            
            resnet18.layer2,
            SelfAttention(128),
            resnet18.layer3,
            SelfAttention(256),  # Adjust based on the output channels of your last layer
            resnet18.layer4,
            
            resnet18.avgpool
        )
        self.fc = resnet18.fc

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ModifiedResNet(in_channels=1, num_classes=7)
model.load_state_dict(torch.load('D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\best_resnet18_model.pth'))
model = model.to(device)
model.eval()  #

# 加载图像并进行预处理
image_path = 'D:\模式识别\Fer2013-Facial-Emotion-Recognition-Pytorch-main\ParrtenRegination\image_comparison_original\original_5.png'
image = Image.open(image_path).convert('L')  # 转为灰度图
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])
input_data = transform(image).unsqueeze(0).to(device)  # 添加批次维度

# 前向传播以获取激活值
with torch.no_grad():
    model(input_data)

# 选择要可视化的层
target_layer = model.features[6]

# 全局作用域使用 global
activation = None
def hook_fn(module, input, output):
    global activation
    activation = output
hook = target_layer.register_forward_hook(hook_fn)

# 前向传播以获取激活值
with torch.no_grad():
    model(input_data)

# 移除hook
hook.remove()

# 获取跨通道的平均激活值
mean_activation = activation.mean(dim=(0, 2, 3)).cpu().numpy()

# 可视化激活图
plt.imshow(mean_activation, cmap='viridis')
plt.colorbar()
plt.show()
