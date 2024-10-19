def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    """
    This function is copied from here
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    This is to ensure that all layers have channels that are divisible by 8.
    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.
    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.append(nn.BatchNorm2d(out_channels))
    if act:
        conv.append(nn.ReLU6())
    return conv


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample,
                 stride, expand_ratio):
        """An inverted bottleneck block with optional depthwises.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        """
        super(UniversalInvertedBottleneckBlock, self).__init__()
        # starting depthwise conv
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv2d(in_channels, in_channels, kernel_size=start_dw_kernel_size, stride=stride_, groups=in_channels, act=False)
        # expansion with 1x1 convs
        expand_filters = make_divisible(in_channels * expand_ratio, 8)
        self._expand_conv = conv2d(in_channels, expand_filters, kernel_size=1)
        # middle depthwise conv
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_filters)
        # projection with 1x1 convs
        self._proj_conv = conv2d(expand_filters, out_channels, kernel_size=1, stride=1, act=False)

        # expand depthwise conv (not used)
        # _end_dw_kernel_size = 0
        # self._end_dw = conv2d(out_channels, out_channels, kernel_size=_end_dw_kernel_size, stride=stride, groups=in_channels, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x

class C2f_UIB(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv3 = UniversalInvertedBottleneckBlock(c2, c2, 5, 3, True, 1, 4)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(self.cv2(torch.cat(y, 1)))

class C3k2_UIB(C2f_UIB):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )