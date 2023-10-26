import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

if torch.cuda.is_available():
    print("CUDA (GPU support) is available and PyTorch can use GPUs!")
else:
    print("CUDA is not available. PyTorch will use CPU.")

# 检查 CUDA 是否可用并定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 数据加载和预处理
class MyDataSet(Dataset):
    def __init__(self, dataset_type, transform=None):
        """
        dataset_type: ['train', 'test']
        """

        dataset_path = 'data/'
        self.transform = transform
        self.sample_list = list()
        self.dataset_type = dataset_type
        f = open(dataset_path + self.dataset_type + '/datalist.txt')
        lines = f.readlines()
        for line in lines:
            self.sample_list.append(line.strip())
        f.close()

    def __getitem__(self, index):
        item = self.sample_list[index]
        img = Image.open(item.split(' ')[0]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = int(item.split(' ')[-1])
        return img, label

    def __len__(self):
        return len(self.sample_list)


transform = transforms.Compose([
    transforms.Resize((227, 227)),  # 放大图片到 227x227
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, ), (0.5, 0.5, 0.5, ))
])

trainset = MyDataSet("train", transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = MyDataSet("test", transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# 定义 AlexNet 模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=9):  # 注意：数据集的类别
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 输入通道为 3，因为数据集是彩色图片
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


model = AlexNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_accuracies = []

# 训练模型
for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(trainloader))
    # 在每个 epoch 结束后计算测试集的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracies.append(100 * correct / total)

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Test Accuracy: {100 * correct / total}%")

# 绘制 trainLoss 和 testAccu
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(train_losses, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Test Accuracy', color=color)
ax2.plot(test_accuracies, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

print("Finished Training")

# 评估模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total}%")
