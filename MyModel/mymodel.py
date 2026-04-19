'''
DONE : cnn kernel size 5x5 to 3x3.
DONE : add residual block
DONE : add batch normalization
TODO : GPU support. (cuda, device, to(device) etc..)
TODO : adjust model fc size.. later  
'''


import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import os

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 256
EPOCH_SIZE = 3
SEED = 245

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
g = torch.Generator()
g.manual_seed(SEED)

# num_workers > 0일경우, windows 상에서 sub프로세스 하나 새로 띄움. 멀티프로세싱 꼬여서 RuntimeERR 발생가능
trainset = torchvision.datasets.CIFAR10(root='./CNN_MyModel/MyModel/dataPT/',train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                          shuffle=True, num_workers=0,
                                          generator=g)

testset = torchvision.datasets.CIFAR10(root='./CNN_MyModel/MyModel/dataPT/',train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
                                          shuffle=False, num_workers=0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                                kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        self.shortcut_bn = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.shortcut_bn(self.shortcut(x)) # outchannels * 32 * 32
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)

class Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, pool_size = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.pool = nn.MaxPool2d(self.pool_size, self.pool_size) # 32->16
        self.res1 = ResidualBlock(in_channels, out_channels) 
        self.res2 = ResidualBlock(out_channels, out_channels) 
        self.fc1 = nn.Linear(out_channels*(32//self.pool_size**2)**2, 120) # if outchannel = 8 && poolsize = 2, Image size is 32x32, res has 2 pools(2,2).
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.res1(x))
        x = self.pool(self.res2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
        
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net.train()

process = psutil.Process(os.getpid())
peak_ram = 0.0

for epoch in range(EPOCH_SIZE):
    delta_time = time.perf_counter()
    total_loss = 0.0
    train_cases = 0

    for inputs, labels in trainloader:
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        train_cases += labels.size(0)
        
        ram = process.memory_info().rss
        peak_ram = max(peak_ram, ram)

    delta_time = time.perf_counter() - delta_time
    print(f'FINISHED EPOCH[{epoch+1}] TRAIN, ELAPSED TIME : [{delta_time*1000:.3f}] ms')
    print(f'THROUGHPUT : [{train_cases/delta_time:.3f}]')
    print(f"TRAIN EPOCH [{epoch+1}] LOSS : {total_loss/train_cases:.3f}")
    # rss output is byte size.
    print(f"PEAK RAM: {peak_ram / (1024**2):.2f} MB")



PATH = './CNN_MyModel/MyModel/dataPT/cifar_net_mymodel.pth'
torch.save(net.state_dict(), PATH)

# EVALUATE
running_loss = 0.0
running_cases = 0
total_loss = 0.0
total_test_cases = 0
correct_cases = 0
net.eval()

with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels = data

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # batch size 별 가중치 부여, batch size보다 작은 loss data는 total loss 평균 계산 보정함
        total_loss += loss.item() * labels.size(0)
        total_test_cases += labels.size(0)
        running_loss += loss.item() * labels.size(0)
        running_cases += labels.size(0)
        preds = outputs.argmax(dim=1)
        correct_cases += (preds == labels).sum().item()
        
        if(i % 50 == 49):
            print(f"TEST LOOP [{i/50}] LOSS : {running_loss/running_cases:.3f}")
            running_loss = 0.0
            running_cases = 0

if(running_cases > 0) :
    print(f"TEST LOOP [LAST] LOSS : {running_loss/running_cases:.3f}")

#맞춘 갯수
print(f"TOTAL ACCURACY : [{correct_cases/total_test_cases*100:.3f}]")
#예상 확률이 정답에 얼마나 멀어졌는가
print(f"TOTAL VERIFY LOSS : [{total_loss/total_test_cases:.3f}]")
print("EVALUATE DONE")
