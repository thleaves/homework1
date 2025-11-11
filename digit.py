import torch
import os
import io
import zipfile
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import datetime


import time

def setup_device():
    """设置设备配置"""
    #print("PyTorch 版本：", torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("设备：", device)
    #print("CUDA 可用：", torch.cuda.is_available())
    #print("cuDNN 已启用：", torch.backends.cudnn.enabled)
    #print("支持的 CUDA 版本：", torch.version.cuda)
    #print("cuDNN 版本：", torch.backends.cudnn.version())
    return device


class MyDataset(Dataset):
    def __init__(self, zip_path, transform=None, preload_to_memory=True):
        self.zip_path = zip_path
        self.transform = transform
        self.preload_to_memory = preload_to_memory

        # 初始化数据结构
        self.samples = []
        self.labels_set = set()
        self.images = [] if preload_to_memory else None
        self.zip_file = None

        # 打开ZIP文件
        self.zip_file = zipfile.ZipFile(zip_path, 'r')

        # 获取ZIP文件中的所有BMP文件
        bmp_files = [f for f in self.zip_file.namelist()
                    if f.lower().endswith('.bmp') and not f.endswith('/')]

        # 第一步：收集所有文件信息和标签
        file_info = []
        for img_path in bmp_files:
            img_name = os.path.basename(img_path)
            # 根据你的文件命名规则提取标签
            # 例如：对于 "0_0.bmp"，提取第一个字符 "0" 作为标签
            first_char = img_name[0]  # 提取文件名第一个字符作为标签
            file_info.append((img_path, first_char))
            self.labels_set.add(first_char)

        # 建立标签映射
        self.classes = sorted(list(self.labels_set))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        # 第二步：根据预加载选项构建样本
        if preload_to_memory:
            self.images = []
            for img_path, label in file_info:
                # 从ZIP文件中读取图像
                with self.zip_file.open(img_path) as f:
                    image_data = f.read()
                    image = Image.open(io.BytesIO(image_data)).convert('L')
                    self.images.append(image)
                    label_idx = self.class_to_idx[label]
                    self.samples.append((len(self.images) - 1, label_idx))
        else:
            for img_path, label in file_info:
                label_idx = self.class_to_idx[label]
                self.samples.append((img_path, label_idx))

        print(f"数据集加载完成：{len(self.samples)} 个样本，{len(self.classes)} 个类别")
        print(f"类别映射：{self.class_to_idx}")

    def __getitem__(self, idx):
        if self.preload_to_memory:
            img_idx, label_idx = self.samples[idx]
            image = self.images[img_idx]
        else:
            img_path, label_idx = self.samples[idx]
            # 从ZIP文件中读取图像
            with self.zip_file.open(img_path) as f:
                image_data = f.read()
                image = Image.open(io.BytesIO(image_data)).convert('L')

        if self.transform:
            image = self.transform(image)
        return image, label_idx

    def __len__(self):
        return len(self.samples)

    def get_original_label(self, idx):
        _, label_idx = self.samples[idx]
        return self.idx_to_class.get(label_idx)

    def close(self):
        """关闭ZIP文件"""
        if self.zip_file:
            self.zip_file.close()

    def __del__(self):
        """析构函数，确保ZIP文件被关闭"""
        self.close()


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.bn2= nn.BatchNorm2d(20)

        self.fc1 = nn.Linear(20 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #x = torch.relu(self.conv1(x))
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        # x = torch.relu(self.conv2(x))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 20 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(train_loader,model, device,  optimizer, epoch,loss_fn):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 99:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1)/ len(train_loader), loss.item()))
            #if args.dry_run:
              #  break


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test result: \n Accuracy:{(100 * correct):>0.1f}%, Avg loss:{test_loss:>8f}")


def main():
    # 设备设置
    device = setup_device()
    torch.manual_seed(42)

    #epochs = 10时效果稳定99.2%~99.3%
    epochs = 5 #Accuracy>=99.2%
    batch_size = 100
    num_workers = 0 # 统一设置工作进程数（必须=0）

    # 数据变换
    transform = transforms.Compose([
        #transforms.Resize((28, 28)),
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 创建数据集和数据加载器（一次性创建）
    train_dataset = MyDataset('data/1-Digit-TrainSet.zip', transform=transform)
    test_dataset = MyDataset('data/1-Digit-TestSet.zip', transform=transform)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"所有标签: {train_dataset.classes}")
    print(f"标签映射: {train_dataset.class_to_idx}")

    # 创建数据加载器（使用相同的num_workers）
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory = True,
        #prefetch_factor= 2,#单线程不可用
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,  # 测试集通常不需要shuffle
        pin_memory=True,
        #prefetch_factor=2,#单线程不可用
        num_workers=num_workers
    )

    # 模型创建和训练
    model = SimpleCNN().to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()

    #optimizer = torch.optim.Adam(model.parameters(), lr= 0.001 )#收敛速度慢
    optimizer = torch.optim.Adadelta(model.parameters(), lr= 1)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.7)

    for epoch in range(1, epochs + 1):
        test(test_dataloader, model, loss_fn, device)
        train(train_dataloader, model, device, optimizer, epoch, loss_fn)
        scheduler.step()
        if epoch % 5 == 0:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), f'models/model_{current_time}_epoch_{epoch}.pt')
            print(f"模型已保存: models/model_{current_time}_epoch_{epoch}.pt")
    test(test_dataloader, model, loss_fn, device)

    print("训练完成!")


if __name__ == "__main__":
    main()