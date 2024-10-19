class SELayer(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SELayer, self).__init__()
        assert reduction >= 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super(DepSepConv, self).__init__()
        assert stride in [1, 2]
        padding = (kernel_size - 1) // 2
        if use_se:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Hardswish(),

                # SE
                SELayer(inp, inp),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.Hardswish(),

            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Hardswish(),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.Hardswish()
            )

    def forward(self, x):
        return self.conv(x)