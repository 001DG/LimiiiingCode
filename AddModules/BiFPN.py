import torch.nn as nn
import torch


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Bi_FPN(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = swish()
        self.epsilon = 0.0001

    def forward(self, x):
        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)  # 权重归一化处理
        weighted_feature_maps = [weights[i] * x[i] for i in range(len(x))]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)
        return result

        # elif m is Bi_FPN:
        #     length = len([ch[x] for x in f])
        #     args = [length]

    # 完整项目文件介绍
    # 下面是大家购买专栏进群内能够获得的文件部分文件截图(CSDN上提供完整文件的本专栏为独一份)，这些代码我已经全部配置好并注册
    # 在模型内大家只需要运行vam文件即可，同时我总结了接近150 + 份的vaml文件组合供大家使用(群内有我的录制的讲解视频，教大家如何
    # 去修改和融合模型)，同时大家也可以自己进行组合，估计组合起来共有上千种，总有一种适合你的数据集Q，让大家成功写出论文。
    # 拥有这个文件YOLOv10你就可以一网打尽，文件均已注册完毕，只许动手点击运行yaml文件即可，非常适合小白。

