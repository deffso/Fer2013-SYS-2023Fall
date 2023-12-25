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
from scipy.stats import poisson
#import cv2
# 添加高斯噪声
def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the image."""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# 添加盐和胡椒噪声
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """Add salt and pepper noise to the image."""
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Add salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 1

    # Add pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# 添加泊松噪声
def add_poisson_noise(image):
    """Add Poisson noise to the image."""
    noisy_image = poisson.rvs(image.astype(np.float32))
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# 添加周期性噪声
def add_periodic_noise(image, frequency=10):
    """Add periodic noise to the image."""
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    noise = 128 * np.sin(2 * np.pi * frequency * y / image.shape[0]) + 128
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


# 添加运动模糊
def add_motion_blur(image, kernel_size=15):
    """Add motion blur to the image."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

# 添加量化噪声
def add_quantization_noise(image, levels=32):
    """Add quantization noise to the image."""
    quantization_levels = np.linspace(0, 255, levels)
    quantized_image = np.digitize(image, quantization_levels) - 1
    noisy_image = quantization_levels[quantized_image]
    return noisy_image.astype(np.uint8)

# 添加散斑噪声
def add_speckle_noise(image, scale=0.1):
    """Add speckle noise to the image."""
    gaussian_noise = np.random.normal(0, scale, image.shape)
    noisy_image = image + image * gaussian_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform, augment=False,noise_type=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.noise_type=noise_type
        self.augment = augment

    def __len__(self):
        return len(self.images)
    
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])
        img = Image.fromarray(img)
        #img = Image.fromarray(img)
        #img = transforms.ToTensor()(img)
        
        if self.noise_type is not None:
            img_array = np.array(img)
            if self.noise_type == 'gaussian':
                img_array = add_gaussian_noise(img_array)
            elif self.noise_type == 'salt_and_pepper':
                img_array = add_salt_and_pepper_noise(img_array)
            elif self.noise_type == 'poisson':
                img_array = add_poisson_noise(img_array)
            elif self.noise_type == 'periodic':
                img_array = add_periodic_noise(img_array)
            elif self.noise_type == 'motion_blur':
                img_array = add_motion_blur(img_array)
            elif self.noise_type == 'quantization':
                img_array = add_quantization_noise(img_array)
            elif self.noise_type == 'speckle':
                img_array = add_speckle_noise(img_array)
            else:
                pass

            img = Image.fromarray(img_array)
        
        
        
        
        
        if self.transform:
            img = self.transform(img)
        # Convert image to tensor
        img = transforms.ToTensor()(img)    
        
        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)
    
        return sample
    
    
def save_images(images, labels, output_folder, prefix):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx in range(min(100, len(images))):
        img = images[idx]
        label = labels[idx]

        img = Image.fromarray(img)
        img = img.convert("RGB")

        img_path = os.path.join(output_folder, f"{prefix}_{idx}.png")
        img.save(img_path)
            

def load_data(path='D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\ParrtenRegination\\fer2013.csv'):
    fer2013 = pd.read_csv(path)
    print(fer2013)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def get_dataloaders(path='D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\ParrtenRegination\\fer2013.csv', bs=128, augment=True,Noise_type=None):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation
        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])
    
    original_output_folder = 'D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\ParrtenRegination\\image_comparison_original'
    #save_images(xtrain, ytrain, original_output_folder, "original")
   
    
    mu, st = 0, 255

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        #transforms.TenCrop(40),
        
        #transforms.Lambda(lambda crops: torch.stack(
          #  [transforms.ToTensor()(crop) for crop in crops])),
        #transforms.Lambda(lambda tensors: torch.stack(
          # [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])
    
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            #transforms.RandomResizedCrop(48, scale=(0.3, 0.7)),
            #transforms.RandomErasing(),
            
            #transforms.FiveCrop(40),
            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            #transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing()(t) for t in tensors])),
        ])
    

    # X = np.vstack((xtrain, xval))
    # Y = np.hstack((ytrain, yval))

    train = CustomDataset(xtrain, ytrain, transform=train_transform,noise_type=Noise_type)
    val = CustomDataset(xval, yval, transform=test_transform,)
    test = CustomDataset(xtest, ytest, transform=test_transform,)

    #train.save_image_comparison('D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\ParrtenRegination\\image_comparison_before_after')
    # 保存增强后的图像
    augmented_output_folder = 'D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\ParrtenRegination\\image_comparison_augmented'
    #save_images(train.images, train.labels, augmented_output_folder, "augmented")
    
    trainloader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
    valloader = DataLoader(val, batch_size=128, shuffle=True, num_workers=0)
    testloader = DataLoader(test, batch_size=128, shuffle=True, num_workers=0)

    return trainloader, valloader, testloader


# 超参数
batch_size = 256
learning_rate = 0.001
epochs = 300
noise_type='speckle'
#gaussian  salt_and_pepper  poisson  periodic motion_blur  quantization  speckle
trainloader, valloader, testloader=get_dataloaders(path='D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\ParrtenRegination\\fer2013.csv',bs=128, augment=True,Noise_type=noise_type)

# 使用GPU（如果可用）
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

resnet18 = ModifiedResNet(in_channels=1, num_classes=7)






print(resnet18)
resnet18 = resnet18.to(device)



# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate, weight_decay=1e-4)  # 添加权重衰减项

# 使用学习率调度器
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

# 创建记录日志的文件
from datetime import datetime

now = datetime.now()
hour=now.hour
month = now.month
day = now.day
log_file_path = 'training_logs_尺寸311_attention_双层_加噪'+str(noise_type)+str(month)+'月'+str(day)+'日'+str(hour)
with open(log_file_path, 'w') as log_file:
    log_file.write("Epoch   Train Loss   Train Acc   Val Loss   Val Acc   PTest Loss   PTest ACC   Duration\n ")

# 训练模型并记录loss和准确度
best_accuracy = 0.0
epoch_duration_last=0
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    resnet18.train()
    

    
    start_time = time.time()  # 记录训练开始时间
    for i, data in enumerate(tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}')):
        images, emotions = data
        images = images.to(device)  # 将输入数据移动到 GPU 上
        emotions = emotions.to(device)  # 将标签数据移动到 GPU 上
        #with autocast():
        #    bs12, ncrops, c, h, w = images.shape
        #    images = images.view(-1, c, h, w)
        outputs = resnet18(images)
        loss = criterion(outputs, emotions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += emotions.size(0)
        correct += (predicted == emotions).sum().item()
    
    end_time = time.time()  # 记录训练结束时间
    epoch_duration = end_time - start_time  # 计算训练时长
    epoch_duration_last +=epoch_duration
    epoch_loss = running_loss / len(trainloader)
    epoch_accuracy = correct / total

    # 更新学习率
    scheduler.step(epoch_loss)

   

    resnet18.eval()  # 切换模型到评估模式

    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader, desc=f'Validation - Epoch {epoch + 1}/{epochs}')):
            images, emotions = data
            images = images.to(device)
            emotions = emotions.to(device)
            outputs = resnet18(images)
            loss = criterion(outputs, emotions)

            test_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_total += emotions.size(0)
            test_correct += (predicted == emotions).sum().item()

    test_epoch_loss = test_running_loss / len(valloader)
    test_epoch_accuracy = test_correct / test_total
    
    
    public_test_running_loss = 0.0
    public_test_correct = 0
    public_test_total = 0
    with torch.no_grad():
    
        for i, data in enumerate(tqdm(testloader, desc=f'PublicTest - Epoch {epoch + 1}/{epochs}')):
            images, emotions = data
            images = images.to(device)
            emotions = emotions.to(device)
            outputs = resnet18(images)
            loss = criterion(outputs, emotions)

            public_test_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            public_test_total += emotions.size(0)
            public_test_correct += (predicted == emotions).sum().item()

    public_test_epoch_loss = public_test_running_loss / len(testloader)
    public_test_epoch_accuracy = public_test_correct / public_test_total
    # 打印并写入日志文件
    log_info = f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy * 100:.2f}%, ' \
               f'Val Loss: {test_epoch_loss:.4f}, Val Acc: {test_epoch_accuracy * 100:.2f}%, ' \
                f'PTest Loss: {public_test_epoch_loss:.4f}, PTest Acc: {public_test_epoch_accuracy * 100:.2f}%, ' \
               f'Duration: {epoch_duration_last:.2f} seconds'
    print(log_info)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{epoch + 1}\t{epoch_loss:.4f}\t{epoch_accuracy * 100:.2f}\t{test_epoch_loss:.4f}\t{test_epoch_accuracy * 100:.2f}\t{public_test_epoch_loss:.4f}\t{public_test_epoch_accuracy * 100:.2f}\t{epoch_duration_last:.2f}\n")
    
    if test_epoch_accuracy > best_accuracy:
        best_accuracy = test_epoch_accuracy
        torch.save(resnet18.state_dict(), 'best_resnet18_model.pth') 
    
# 保存模型
print(f'Best Accuracy on Test Set: {best_accuracy * 100:.2f}%')

