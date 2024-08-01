import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.tensorboard import SummaryWriter


# 残差结构（18层）
# 2个3*3的卷积核（步长分别都是1） 卷积核的个数都是64

class BasicBlock(nn.Module):

    expansion = 1  # 残差结构中主分支的卷积核的个数有没有发生变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # 输入64*16*16
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 输出64*16*16
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 下采样函数捷径分支

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,  # 残差结构————basicblock
                 blocks_num,  # 残差结构的数目
                 output=1,  # 训练集的分类个数
                 include_top=True,  # 搭建更加复杂的网络
                 ):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 通过maxpoling得到的矩阵的通道数 残差网络部分的输入通道数

        # 第一层
        # 输入3*64*64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        # 64*32*32
        # 第一层卷积层  第一个3对应输入通道数  in_channel其实是下一步的输入通道的个数也是本层的输出通道的个数
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 第二层最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 输出 64*16*16
        # 残差结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc1 = nn.Linear(512 * block.expansion, 256)  # 对应全连接层输出的分类个数
            self.fc2 = nn.Linear(256, output)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):  # channel:对应第一层残差结构的卷积核的个数 block_num:对应的残差结构的个数
        downsample = None  # 下采样
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,

                            ))
        self.in_channel = channel * block.expansion  # 64

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.fc2(x)
        return x


def resnet18(output=1, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 2, 2, 2], output=output, include_top=include_top)


net = resnet18()


def mean_squared_loss(y_predict, y_true):
    """
    均方误差损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值
    :return:
    """
    # loss, dy = mean_squared_loss(y, y_true)
    k = y_predict.float()
    y_true = y_true

    loss = np.mean(np.sum(np.square(y_predict - y_true), axis=-1))  # 损失函数值
    return loss


#  准备数据集
data = pd.read_csv('multime_data/vcm_main_final_change_data.csv')
data = np.array(data)
data = data.astype(float)
main_data = torch.FloatTensor(data)
# 训练集
train_data = main_data[0:12288]  # 训练数据
train_data = train_data.reshape((3, 64, 64))  # 输入大小1*3*64*64 batch_size=1
train_data = torch.tensor(train_data)
train_data = train_data.unsqueeze(0)  # 增加一个维度
# 测试集
test_data = data[5000:6024]
test_data = test_data.reshape((1, 32, 32))
test_data = torch.tensor(test_data)
test_data = test_data.unsqueeze(0)  # 增加一个维度

target = torch.tensor([0.17])
criterion = nn.MSELoss()
print(net)
optimizer = optim.Adam(net.parameters(), lr=0.0000001)  # -7
loss_list = []

# summary_writer = SummaryWriter("./logs/")
for i in range(1000):
    optimizer.zero_grad()  # 清空梯度
    output = net(train_data)
    # print(output.shape)
    loss = criterion(output[0], target)
    # summary_writer.add_scalar("loss", loss, (i + 1))
    print(loss)
    loss.backward()  # 反向传播计算当前梯度
    loss_list.append(loss)
    optimizer.step()  # 更新网络参数
# summary_writer.close() ccv
print(net)

if __name__ == '__main__':
    for i in range(1000):
        loss_list[i] = loss_list[i].item()
    print(loss_list)
    # plt.figure(figsize=(v10, v10))
    plt.title('Loss', fontsize=17)
    plt.xlabel('per 1000times', fontsize=17)
    plt.ylabel('Loss', fontsize=17)
    # plt.xticks(fontsize=20) #x轴刻度的字体大小（文本包含在pd_data中了）
    # plt.yticks(fontsize=20) #y轴刻度的字体大小（文本包含在pd_data中了）
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.plot(range(1000), loss_list)
    plt.savefig('loss_resnet_final.png')
    plt.show()
