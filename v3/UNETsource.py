import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F

########################################### modern UNet code: ##################################


"""
Modern UNet implementation
Largely based on / extended from
https://github.com/microsoft/pdearena/blob/db7664bb8ba1fe6ec3217e4079979a5e4f800151/pdearena/modules/conditioned/twod_unet.py
which is largely based on
https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/05321d644e4fed67d8b2856adc2f8585e79dfbee/labml_nn/diffusion/ddpm/unet.py
Implemented in 1, 2 and 3 dimensions. Conditioning implemented by broadcasting and concatenating the 0d signal along
hidden features in both the encoder and decoder
"""


class UNetModern(nn.Module):


    """Modern U-Net architecture
    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks
    Args:
        num_spatial_dims (int): Number of spatial dimensions
        n_cond (int): Dimensionality of conditioning signal
        hidden_features (int): Number of channels in the hidden layers
        cond_mode (str): Type of conditioning to apply
        activation (nn.Module): Activation function to use
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers
    """

    def __init__(
        self,
        num_spatial_dims: int = 2,
        hidden_features: int = 128,
        activation: nn.Module = nn.GELU(),
        norm: bool = False,
        ch_mults=(2, 2, 2, 2),
        is_attn=(False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use1x1: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.num_spatial_dims = num_spatial_dims

        self.activation: nn.Module = activation

        # Number of resolutions
        n_resolutions = len(ch_mults)
        n_channels = hidden_features

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        num_spatial_dims=num_spatial_dims,
                        **kwargs
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels, num_spatial_dims=num_spatial_dims, **kwargs))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(in_channels=out_channels, out_channels=out_channels, has_attn=mid_attn,
                                  activation=activation, norm=norm, num_spatial_dims=num_spatial_dims, **kwargs)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        num_spatial_dims=num_spatial_dims,
                        **kwargs
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, has_attn=is_attn[i], activation=activation, norm=norm, num_spatial_dims=num_spatial_dims, **kwargs))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels, num_spatial_dims=num_spatial_dims, **kwargs))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()

        if use1x1:
            self.final = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=hidden_features, out_channels=hidden_features, kernel_size=1, **kwargs)
        else:
            self.final = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, padding=1, **kwargs)



    def _crop_Nd(self, enc_ftrs: torch.Tensor, shape: torch.Tensor):
        if isinstance(shape, torch.Tensor) or isinstance(shape, np.ndarray):
            shape = shape.shape
        s_des = shape[-self.num_spatial_dims:]
        s_current = enc_ftrs.shape[-self.num_spatial_dims:]
        # first, calculate preliminary paddings - may contain non-integers ending in .5):
        pad_temp = np.repeat(np.subtract(s_des, s_current) / 2, 2)
        # to break the .5 symmetry to round one padding up and one down, we add a small pos/neg number respectively
        # note this will not impact the case where pad_temp[i] is integer since it is still rounded to that integer
        breaking_arr = np.tile([1, -1], int(len(pad_temp) / 2)) / 1000
        pad = tuple(map(lambda p: int(round(p)), pad_temp + breaking_arr))
        enc_ftrs = F.pad(enc_ftrs, pad)
        return enc_ftrs

    def forward(self, h: torch.Tensor, variables: torch.Tensor = None, **kwargs):
        assert h.dim() == 2 + self.num_spatial_dims  # [b, c, *spatial_dims]
        h_shape = h.shape
        h_features = [h]
        for m in self.down:
            h = m(h)
            h_features.append(h)

        h = self.middle(h)

        for m in self.up:
            if isinstance(m, Upsample):
                h = m(h)
            else:
                s = self._crop_Nd(h_features.pop(), h)  # crop spatial dim to match features
                # Get the skip connection from first half of U-Net and concatenate
                h = torch.cat((h, s), dim=1)
                h = m(h)

        h = self.final(self.activation(self.norm(h)))
        h = self._crop_Nd(h, h_shape)  # crop spatial dim to match features
        return h


class ResidualBlock(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (nn.Module): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module = torch.nn.GELU(),
        norm: bool = False,
        n_groups: int = 1,
        num_spatial_dims: int = 1,
            **kwargs
    ):
        super().__init__()
        self.activation: nn.Module = activation

        self.conv1 = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, **kwargs)
        self.conv2 = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, **kwargs)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=1, **kwargs)
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        # Second convolution layer
        h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention]
    Args:
        in_channels (int): the number of channels in the input
        n_heads (int): the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups (int): the number of groups for [group normalization][torch.nn.GroupNorm].
    """

    def __init__(self, in_channels: int, out_channels: int = None, n_heads: int = 1, d_k=None, n_groups: int = 1, num_spatial_dims: int = 1, **kwargs):
        super().__init__()

        # Default `d_k`
        if out_channels is None:
            out_channels = in_channels
        if d_k is None:
            d_k = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, in_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(in_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, out_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

        if in_channels != out_channels:
            self.shortcut = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=1, **kwargs)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size, _, *spatial_dims = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, out_channels]`
        res = self.output(res)

        # Add skip connection
        res += self.shortcut(x)

        # Change to shape `[batch_size, out_channels, *spatial_dims]`
        res = res.permute(0, 2, 1).view(batch_size, self.out_channels, *spatial_dims)
        return res


class DownBlock(nn.Module):
    """Down block. This combines ResidualBlock and AttentionBlock.
    These are used in the first half of U-Net at each resolution.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function to use.
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: nn.Module = nn.GELU(),
        norm: bool = False,
        num_spatial_dims: int = 1,
            **kwargs
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, activation=activation, norm=norm,
                                 num_spatial_dims=num_spatial_dims, **kwargs)
        if has_attn:
            self.attn = AttentionBlock(out_channels, num_spatial_dims=num_spatial_dims, **kwargs)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Up block that combines ResidualBlock and AttentionBlock.
    These are used in the second half of U-Net at each resolution.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function to use.
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: nn.Module = nn.GELU(),
        norm: bool = False,
        num_spatial_dims: int = 1,
            **kwargs
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, activation=activation, norm=norm, num_spatial_dims=num_spatial_dims, **kwargs)
        if has_attn:
            self.attn = AttentionBlock(out_channels, num_spatial_dims=num_spatial_dims, **kwargs)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (nn.Module): Activation function to use.
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(self, in_channels, out_channels: int, has_attn: bool = False, activation: nn.Module = nn.GELU(), norm: bool = False,
                 num_spatial_dims: int = 1, **kwargs):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, activation=activation, norm=norm,
                                  num_spatial_dims=num_spatial_dims, **kwargs)
        self.attn = AttentionBlock(out_channels, num_spatial_dims=num_spatial_dims, **kwargs) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(out_channels, out_channels, activation=activation, norm=norm,
                                  num_spatial_dims=num_spatial_dims, **kwargs)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$
    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int, num_spatial_dims: int, **kwargs):
        super().__init__()
        self.conv = get_upconv_with_right_spatial_dim(num_spatial_dims, in_channels=n_channels, out_channels=n_channels, kernel_size=4, stride=2, padding=1, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$
    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int, num_spatial_dims: int, **kwargs):
        super().__init__()
        self.conv = get_conv_with_right_spatial_dim(num_spatial_dims, in_channels=n_channels, out_channels=n_channels, kernel_size=2, stride=2, padding=1, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.conv(x)






########################## some utility functions to get conv layers with the right spatial dims ###################
def get_conv_with_right_spatial_dim(spatial_dim, **kwargs):
    if spatial_dim == 1:
        conv = nn.Conv1d(**kwargs)
    elif spatial_dim == 2:
        conv = nn.Conv2d(**kwargs)
    elif spatial_dim == 3:
        conv = nn.Conv3d(**kwargs)
    else:
        raise NotImplementedError(f'only 0<x<=3d convs implemented so far, but found spatial dim {spatial_dim}!')

    return conv


def get_upconv_with_right_spatial_dim(spatial_dim, in_channels, out_channels, **kwargs):
    kwargs_copy = kwargs
    if 'padding_mode' in kwargs:
        kwargs_copy = {}
        for k, v in kwargs.items():
            if k == 'padding_mode':
                kwargs_copy[k] = 'zeros'
            else:
                kwargs_copy[k] = v

    if spatial_dim == 1:
        upconv = nn.ConvTranspose1d(in_channels, out_channels, **kwargs_copy)
    elif spatial_dim == 2:
        upconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs_copy)
    elif spatial_dim == 3:
        upconv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs_copy)
    else:
        raise NotImplementedError(f'only 0<x<=3d convs implemented so far, but found spatial dim {spatial_dim}!')

    return upconv