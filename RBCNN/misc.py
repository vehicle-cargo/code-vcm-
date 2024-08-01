from torch import nn


class ModuleWrapper(nn.Module):
    def __int__(self):
        super(ModuleWrapper).__int__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():  # 子模块
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():  # 对于第一代模块
            x = module(x)

        kl = 0.0
        for module in self.modules():  # 对于每一个模块
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        return x, kl


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)
