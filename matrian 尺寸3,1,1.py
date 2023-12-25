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
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform

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


def get_dataloaders(path='D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\ParrtenRegination\\fer2013.csv', bs=128, augment=True):
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

    train = CustomDataset(xtrain, ytrain, transform=train_transform,)
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


trainloader, valloader, testloader=get_dataloaders(path='D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\ParrtenRegination\\fer2013.csv',bs=128, augment=True)

# 使用GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
print(device)


resnet18 = models.resnet18(pretrained=True)
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 7)  # 修改最后一层
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
log_file_path = 'training_logs_尺寸311'+str(month)+'月'+str(day)+'日'+str(hour)
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

