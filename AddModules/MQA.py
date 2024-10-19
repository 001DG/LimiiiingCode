
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.append(nn.BatchNorm2d(out_channels))
    if act:
        conv.append(nn.ReLU6())
    return conv

class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides, dw_kernel_size=3, dropout=0.0):
        """Multi Query Attention with spatial downsampling.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        3 parameters are introduced for the spatial downsampling:
        1. kv_strides: downsampling factor on Key and Values only.
        2. query_h_strides: vertical strides on Query only.
        3. query_w_strides: horizontal strides on Query only.

        This is an optimized version.
        1. Projections in Attention is explict written out as 1x1 Conv2D.
        2. Additional reshapes are introduced to bring a up to 3x speed up.
        """
        super(MultiQueryAttentionLayerWithDownSampling, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = self.key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(in_channels)
        self._query_proj = conv2d(in_channels, self.num_heads * self.key_dim, 1, 1, norm=False, act=False)

        if self.kv_strides > 1:
            self._key_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
                                       norm=True, act=False)
            self._value_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
                                         norm=True, act=False)
        self._key_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
        self._value_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
        self._output_proj = conv2d(num_heads * key_dim, in_channels, 1, 1, norm=False, act=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        bs, seq_len, _, _ = x.size()
        # print(x.size())
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = F.avg_pool2d(self.query_h_strides, self.query_w_strides)
            q = self._query_downsampling_norm(q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        px = q.size(2)
        q = q.view(bs, self.num_heads, -1, self.key_dim)  # [batch_size, num_heads, seq_len, key_dim]

        if self.kv_strides > 1:
            k = self._key_dw_conv(x)
            k = self._key_proj(k)
            v = self._value_dw_conv(x)
            v = self._value_proj(v)
        else:
            k = self._key_proj(x)
            v = self._value_proj(x)
        k = k.view(bs, 1, self.key_dim, -1)   # [batch_size, 1, key_dim, seq_length]
        v = v.view(bs, 1, -1, self.key_dim)    # [batch_size, 1, seq_length, key_dim]

        # calculate attention score
        # print(q.shape, k.shape, v.shape)
        attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, dim=-1)

        # context = torch.einsum('bnhm,bmv->bnhv', attn_score, v)
        # print(attn_score.shape, v.shape)
        context = torch.matmul(attn_score, v)
        context = context.view(bs, self.num_heads * self.key_dim, px, px)
        output = self._output_proj(context)
        # print(output.shape)
        return output



class C2f_MQA(nn.Module):
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
        self.att = MultiQueryAttentionLayerWithDownSampling(c2, 8, 48, 48, 1, 1, 1)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))



class C3k2_MQA(C2f_MQA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )