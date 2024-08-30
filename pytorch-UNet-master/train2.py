import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import cv2
import tqdm

# 设置可见的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个 GPU

# 定义 UNet 模型
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        return self.out(O4)

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)
    def forward(self, x, feature_map):
        up = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])


class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.names = os.listdir(os.path.join(path, "train_annotation"))
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        segment_name = self.names[index]
        segment_image_path = os.path.join(self.path, "train_annotation", segment_name)
        image_path = os.path.join(self.path, 'train', segment_name)

        segment_image = cv2.imread(segment_image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path)

        assert image.shape[:2] == segment_image.shape[:2], "Image and segment image size do not match"

        # 对图像应用转换
        if self.transform:
            image = self.transform(image)

        # 对分割图像应用转换
        if self.transform:
            segment_image = self.transform(segment_image)

        # 转换为 PyTorch 张量，并处理为长整型
        segment_image = segment_image.clone().detach().long().unsqueeze(0)

        return image, segment_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(num_classes=2).to(device)

opt = optim.Adam(net.parameters())
loss_fun = nn.CrossEntropyLoss()

data_loader = DataLoader(MyDataset('D:/Users/hejie/Desktop/dataset/train', transform=transform), batch_size=4, shuffle=True)

epoch = 1
while epoch < 200:
    for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
        image, segment_image = image.to(device), segment_image.to(device)

        # 清理未使用的内存
        torch.cuda.empty_cache()

        out_image = net(image)
        train_loss = loss_fun(out_image, segment_image)
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        if i % 1 == 0:
            print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

        _image = image[0]
        _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
        _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

        img = torch.stack([_segment_image, _out_image], dim=0)
        save_image(img, f'{save_path}/{i}.png')
    if epoch % 20 == 0:
        torch.save(net.state_dict(), f'{save_path}/unet_epoch_{epoch}.pth')
        print('save successfully!')
    epoch += 1
