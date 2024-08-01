import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import Parameter
from RBCNN.metrics import calculate_kl as kl_div
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import RBCNN.config as cfg
from RBCNN.misc import ModuleWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1  # 控制卷积核的分组
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1)
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']
        # 定义先验分布的超参数，训练过程中不会发生变化
        self.w_mu = Parameter(
            torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))  # 返回未初始化的张量
        self.w_rho = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.w_mu.data.normal_(*self.posterior_mu_initial)
        self.w_rho.data.normal_(*self.posterior_rho_initial)
        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    # 采样权重和偏置
    def forward(self, input, sample=True):
        if self.training or sample:
            w_exp = torch.empty(self.w_mu.size()).normal_(0, 1).to(self.device)
            # 64*1*v10*v10
            w_exp
            self.w_sigma = torch.log1p(torch.exp(self.w_rho))

            self.w_sigma
            weight = self.w_mu + w_exp * self.w_sigma
            weight
            if self.use_bias:
                bias_exp = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_exp * self.bias_sigma

            else:
                bias = None
        else:
            weight = self.w_mu
            bias = self.bias_mu if self.use_bias else None
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        kl = kl_div(self.prior_mu, self.prior_sigma, self.w_mu, self.w_sigma)
        if self.use_bias:
            kl += kl_div(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)

        return kl


class BBBLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),

            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']
        self.w_mu = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.w_rho = Parameter(torch.empty((out_features, in_features), device=self.device))
        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features), device=self.device))

        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.w_mu.data.normal_(*self.posterior_mu_initial)
        self.w_rho.data.normal_(*self.posterior_rho_initial)
        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):
        if self.training or sample:
            w_eps = torch.empty(self.w_mu.size()).normal_(0, 1).to(self.device)
            self.w_sigma = torch.log1p(torch.exp(self.w_rho))
            weight = self.w_mu + w_eps * self.w_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.w_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(x, weight, bias)

    def kl_loss(self):
        kl = kl_div(self.prior_mu, self.prior_sigma, self.w_mu, self.w_sigma)
        if self.use_bias:
            kl += kl_div(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)

        return kl


# 残差结构（18层）
# 2个3*3的卷积核（步长分别都是1） 卷积核的个数都是64

class BasicBlock(ModuleWrapper):
    expansion = 1  # 残差结构中主分支的卷积核的个数有没有发生变化

    def __init__(self, priors, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.priors = priors
        self.priors = cfg.priors

        # 输入64*16*16
        self.conv1 = BBBConv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=True, priors=self.priors)
        # 输出64*16*16
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = BBBConv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
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


class ResNet(ModuleWrapper):

    def __init__(self,
                 priors,
                 block,  # 残差结构————basicblock
                 blocks_num,  # 残差结构的数目
                 output=1,  # 训练集最终输出
                 include_top=True,
                 # 搭建更加复杂的网络
                 ):
        super(ResNet, self).__init__()
        self.priors = priors
        self.priors = cfg.priors
        self.include_top = include_top
        self.in_channel = 64  # 通过maxpoling得到的矩阵的通道数 残差网络部分的输入通道数

        # 第一层
        # 输入3*64*64
        self.conv1 = BBBConv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=True, priors=self.priors)
        # 64*32*32
        # 第一层卷积层  第一个3对应输入通道数  in_channel其实是下一步的输入通道的个数也是本层的输出通道的个数
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 第二层最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 输出 64*16*16
        # 残差结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        #self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc1 = BBBLinear(64 * block.expansion, 32, bias=True, priors=self.priors)  # 对应全连接层输出的分类个数
            self.fc2 = BBBLinear(32, 1, bias=True, priors=self.priors)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 对卷积神经网络得权重参数进行初始化

    def _make_layer(self, block, channel, block_num, stride=1):  # channel:对应第一层残差结构的卷积核的个数 block_num:对应的残差结构的个数
        downsample = None  # 下采样
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                BBBConv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=True,
                          priors=self.priors),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.priors,
                            self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride

                            ))
        self.in_channel = channel * block.expansion  # 64

        for _ in range(1, block_num):
            layers.append(block(self.priors,
                                self.in_channel,
                                channel
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        # x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.fc2(x)
        return x


priors = cfg.priors


def resnet18(output=1, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(priors, BasicBlock, [3, 2, 2, 2], output=output, include_top=include_top)


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
data = pd.read_csv('kpca_degree.csv')
# 计算最小-最大归一化参数
min_val = data.min().min()  # 找到所有数据中的最小值
max_val = data.max().max()  # 找到所有数据中的最大值

# 应用归一化到整个数据集
normalized_data = (data - min_val) / (max_val - min_val)
data = np.array(normalized_data)
data = data.astype(float)
main_data = torch.FloatTensor(data).to(device)
# 训练集
train_data = main_data[0:12288]  # 训练数据
train_data = train_data.reshape((3, 64, 64))  # 输入大小1*3*64*64 batch_size=1
train_data = torch.tensor(train_data)
train_data = train_data.unsqueeze(0).to(device)  # 增加一个维度
# 加载新的测试数据并进行预测
test_data = normalized_data[16812:29100]
data = np.array(test_data)
main_data = data.astype(float)
main_data = torch.FloatTensor(main_data).to(device)
new_test_data = main_data.reshape((3, 64, 64))
new_test_data = torch.tensor(new_test_data)
new_test_data = new_test_data.unsqueeze(0).to(device)

target = torch.tensor([[0.35]]).to(device)
criterion = nn.MSELoss().to(device)
net = resnet18(priors).to(device)
optimizer = optim.SGD(net.parameters(), lr=0.0001)  # 0.00002
train_loss_list = []
test_loss_list = []
# summary_writer = SummaryWriter(log_dir="../logs")
for i in range(500):
    net.train()
    optimizer.zero_grad()  # 清空梯度
    output = net(train_data).to(device)
    print(output.shape)
    loss = criterion(output, target).to(device)
    #  summary_writer.add_scalar("loss", loss, (i + 1))
    print(loss)
    loss.backward()  # 反向传播计算当前梯度-
    train_loss_list.append(loss)
    optimizer.step()  # 更新网络参数
    # 进行预测
    net.eval()
    with torch.no_grad():
        test_output = net(new_test_data).to(device)
        test_loss = criterion(test_output, target).to(device)
        print(test_loss)
        test_loss_list.append(test_loss.item())
# summary_writer.close()
print(net)
torch.save(net.state_dict(), 'model_weights.pth')


#
# # 加载模型
# net = resnet18(priors).to(device)
# net.load_state_dict(torch.load('model_weights.pth'))


if __name__ == '__main__':
    for i in range(500):
        train_loss_list[i] = train_loss_list[i].item()
        test_loss_list[i] = test_loss_list[i].item()
    print(train_loss_list)
    print(test_loss_list)
    # plt.figure(figsize=(v10, v10))
    plt.title('Loss', fontsize=17)
    plt.xlabel('per 500times', fontsize=17)
    plt.ylabel('Loss', fontsize=17)
    # plt.xticks(fontsize=20) #x轴刻度的字体大小（文本包含在pd_data中了）
    # plt.yticks(fontsize=20) #y轴刻度的字体大小（文本包含在pd_data中了）
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.plot(range(500), train_loss_list,label='Train loss')
    plt.plot(range(500), test_loss_list, label='test loss')
    plt.savefig('loss_final.png')
    plt.show()
