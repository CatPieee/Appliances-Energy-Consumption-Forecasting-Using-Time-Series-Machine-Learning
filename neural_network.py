import torch #手电筒
from torch import nn    # neural network
from torch.utils.data import Dataset, DataLoader   # 数据加载器 ≈ pandas 数据集加载，预处理，划分
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_dataset = pd.read_csv('data/energydata_complete.csv')
print("数据集的样本尺寸：", df_dataset.shape) # (19735, 29)
X = df_dataset.iloc[:, 2:-1]  # 特征
y = df_dataset.iloc[:, 1]  # 目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 数据转tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):          # index 索引
        return self.X[idx], self.y[idx]

train_dataset = MyDataset(X_train_tensor, y_train_tensor)
test_dataset = MyDataset(X_test_tensor, y_test_tensor)
print("训练集尺寸: ", train_dataset.__len__()) # torch.Size([64, 26])
print("测试集尺寸: ", test_dataset.__len__())  # torch.Size([64])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for x, y in train_loader:
    print("Shape of x [N, feature]: ", x.shape)
    print("y: ", y.shape)
    print("查看第一个数据：")
    print("x: ", x[0])
    print("y: ", y[0])
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # 计算设备，用cpu 还是gpu

# Define model
class MyNeuralNetwork(nn.Module): # 父类，继承nn.Module 才能pytorch的功能
    def __init__(self):  #初始化方法
        super().__init__()  # super是调用父类的方法（函数）
        # 准备神经网络的各个层
        #self.flatten = nn.Flatten()  # 创造一个类的属性，是nn的Flatten 展开层  把后面的赋值给左边 后面的是电筒🔦的方法，直接调用，标准化方法（只有图像需要）
        self.linear_relu_stack = nn.Sequential( # nn.Sequential()用于把一组神经网络层按顺序连接
            nn.Linear(26, 10),         # 线性神经网络  output 随机
            nn.ReLU(),                                 # 非线形激活函数
            nn.Linear(10, 1), # 线性神经网络
            #nn.ReLU(),                                 #非线形激活函数
            #nn.Linear()  # 线性神经网络
        )

    def forward(self, x): # 父类方法，重写 实现自己的身形网络，前向传播 self = 是这个class的方法（函数）调用

        # x = self.flatten(x)  #展平，只有图像需要
        #print("展开后数据的形状",x.shape)
        y = self.linear_relu_stack(x) # torch.Size([64, 1])
        #print("输出时数据的形状",logits.shape)
        return y.squeeze(1) # torch.Size([64])





model = MyNeuralNetwork().to(device)
#print(model)

#loss_fn = nn.CrossEntropyLoss()  # 损失函数 nn调用电筒 ， 交叉熵损失函数 # 用于分类问题的，离散的

loss_fn = nn.MSELoss() #用于回归，连续值问题

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # 优化器，电筒的默认语法：创建一个优化器 SGD优化器 # learn rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test set: Average loss (MSE): {test_loss:>8f}")

    # 可选：计算并显示其他回归指标，如MAE
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_preds.append(pred)
            all_targets.append(y)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    print(f"Test set: MAE: {mae:>8f}")

# 需要训练的是权重矩阵

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)

print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
