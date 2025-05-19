"""
Authors:
    - Francesco Paissan, 2023
    - Alberto Ancilotto, 2023
    - Riccardo Benevelli, 2025
"""
import torch
import torch.nn as nn

from typing import Union, Tuple, Optional, List, Dict


def autopad(k: int, p: Optional[int] = None):
    """Implements padding to mimic "same" behaviour.
    Arguments
    ---------
    k : int
        Kernel size for the convolution.
    p : Optional[int]
        Padding value to be applied.

    """
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class XiConv(nn.Module):
    """Implements XiNet's convolutional block as presented in the original paper.

    Arguments
    ---------
    c_in: int
        Number of input channels.
    c_out: int
        Number of output channels.
    kernel_size: Union[int, Tuple]
        Kernel size for the main convolution.
    stride: Union[int, Tuple]
        Stride for the main convolution.
    padding: Optional[Union[int, Tuple]]
        Padding that is applied in the main convolution.
    groups: Optional[int]
        Number of groups for the main convolution.
    act: Optional[bool]
        When True, uses SiLU activation function after
        the main convolution.
    gamma: Optional[float]
        Compression factor for the convolutional block.
    attention: Optional[bool]
        When True, uses attention.
    skip_tensor_in: Optional[bool]
        When True, defines broadcasting skip connection block.
    skip_res : Optional[List]
        Spatial resolution of the skip connection, such that
        average pooling is statically defined.
    skip_channels: Optional[int]
        Number of channels for the input block.
    pool: Optional[bool]
        When True, applies pooling after the main convolution.
    attention_k: Optional[int]
        Kernel for the attention module.
    attention_lite: Optional[bool]
        When True, uses efficient attention implementation.
    batchnorm: Optional[bool]
        When True, uses batch normalization inside the ConvBlock.
    dropout_rate: Optional[int]
        Dropout probability.
    skip_k: Optional[int]
        Kernel for the broadcast skip connection.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: Union[int, Tuple] = 3,
        stride: Union[int, Tuple] = 1,
        padding: Optional[Union[int, Tuple]] = None,
        groups: Optional[int] = 1,
        act: Optional[bool] = True,
        gamma: Optional[float] = 4,
        attention: Optional[bool] = True,
        skip_tensor_in: Optional[bool] = True,
        skip_res: Optional[List] = None,
        skip_channels: Optional[int] = 1,
        pool: Optional[int] = None,
        attention_k: Optional[int] = 3,
        attention_lite: Optional[bool] = True,
        batchnorm: Optional[bool] = True,
        dropout_rate: Optional[int] = 0,
        skip_k: Optional[int] = 1,
    ):
        super().__init__()
        self.compression = int(gamma)
        self.attention = attention
        self.attention_lite = attention_lite
        self.attention_lite_ch_in = c_out // self.compression // 2
        self.pool = pool
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate

        if skip_tensor_in:
            assert skip_res is not None, "Specifcy shape of skip tensor."
            self.adaptive_pooling = nn.AdaptiveAvgPool2d(
                (int(skip_res[0]), int(skip_res[1]))
            )

        if self.compression > 1:
            self.compression_conv = nn.Conv2d(
                c_in, c_out // self.compression, 1, 1, groups=groups, bias=False
            )
        self.main_conv = nn.Conv2d(
            c_out // self.compression if self.compression > 1 else c_in,
            c_out,
            kernel_size,
            stride,
            groups=groups,
            padding=autopad(kernel_size, padding),
            bias=False,
        )
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

        if attention:
            if attention_lite:
                self.att_pw_conv = nn.Conv2d(
                    c_out, self.attention_lite_ch_in, 1, 1, groups=groups, bias=False
                )
            self.att_conv = nn.Conv2d(
                c_out if not attention_lite else self.attention_lite_ch_in,
                c_out,
                attention_k,
                1,
                groups=groups,
                padding=autopad(attention_k, None),
                bias=False,
            )
            self.att_act = nn.Sigmoid()

        if pool:
            self.mp = nn.MaxPool2d(pool)
        if skip_tensor_in:
            self.skip_conv = nn.Conv2d(
                skip_channels,
                c_out // self.compression,
                skip_k,
                1,
                groups=groups,
                padding=autopad(skip_k, None),
                bias=False,
            )
        if batchnorm:
            self.bn = nn.BatchNorm2d(c_out)
        if dropout_rate > 0:
            self.do = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """Computes the forward step of the XiNet's convolutional block.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        ConvBlock output. : torch.Tensor
        """
        s = None
        # skip connection
        if isinstance(x, list):
            # s = F.adaptive_avg_pool2d(x[1], output_size=x[0].shape[2:])
            s = self.adaptive_pooling(x[1])
            s = self.skip_conv(s)
            x = x[0]

        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)

        if s is not None:
            x = x + s

        if self.pool:
            x = self.mp(x)

        # main conv and activation
        x = self.main_conv(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.act(x)

        # attention conv
        if self.attention:
            if self.attention_lite:
                att_in = self.att_pw_conv(x)
            else:
                att_in = x
            y = self.att_act(self.att_conv(att_in))
            x = x * y

        if self.dropout_rate > 0:
            x = self.do(x)

        return x


class XiNet(nn.Module):
    """Defines a XiNet.

    Arguments
    ---------
    input_shape : List
        Shape of the input tensor.
    alpha: float
        Width multiplier.
    gamma : float
        Compression factor.
    num_layers : int = 5
        Number of convolutional blocks.
    num_classes : int
        Number of classes. It is used only when include_top is True.
    include_top : Optional[bool]
        When True, defines an MLP for classification.
    base_filters : int
        Number of base filters for the ConvBlock.
    min_feature_size : Optional[int]
        Minimum feature map size before stopping downsampling. Default is 7.
    return_blocks : Optional[List]
        List of block indices to return.
        Each block consists of a downsampling layer (if needed) and a regular layer.

    Example
    -------
    .. doctest::

        >>> from micromind.networks import XiNet
        >>> model = XiNet((3, 224, 224))
    """

    def __init__(
        self,
        input_shape: List,
        alpha: float = 1.0,
        gamma: float = 4.0,
        num_layers: int = 5,
        num_classes=1000,
        include_top=False,
        base_filters: int = 16,
        min_feature_size: Optional[int] = 7,
        return_blocks: Optional[List] = None,
    ):
        super().__init__()

        self._layers = nn.ModuleList([])
        self.input_shape = torch.Tensor(input_shape)
        self.include_top = include_top
        self.return_blocks = return_blocks
        
        self.block_to_layer_map = {}
        self.layer_to_block_map = {}
        
        count_downsample = 0
        current_layer_idx = 0
        current_block_idx = 0

        feature_size = [input_shape[1] // 2, input_shape[2] // 2]

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_shape[0],
                int(base_filters * alpha),
                7,
                padding=7 // 2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(int(base_filters * alpha)),
            nn.SiLU(),
        )
        count_downsample += 1

        num_filters = [
            int(2 ** (base_filters**0.5 + i)) for i in range(0, num_layers)
        ]
        skip_channels_num = int(base_filters * 2 * alpha)

        for i in range(len(num_filters) - 2):  # Account for the last two layers separately
            # Track the block index for this layer
            self.block_to_layer_map[current_block_idx] = current_layer_idx
            
            need_downsampling = min(feature_size) > min_feature_size
            
            if need_downsampling:
                self._layers.append(
                    XiConv(
                        int(num_filters[i] * alpha),
                        int(num_filters[i + 1] * alpha),
                        kernel_size=3,
                        stride=1,
                        pool=2,
                        skip_tensor_in=(i != 0),
                        skip_res=self.input_shape[1:] / (2**count_downsample),
                        skip_channels=skip_channels_num,
                        gamma=gamma,
                    )
                )
                self.layer_to_block_map[current_layer_idx] = current_block_idx
                current_layer_idx += 1
                count_downsample += 1
                feature_size = [fs // 2 for fs in feature_size]
            else:
                self._layers.append(
                    XiConv(
                        int(num_filters[i] * alpha),
                        int(num_filters[i + 1] * alpha),
                        kernel_size=3,
                        stride=1,
                        pool=None,
                        skip_tensor_in=(i != 0),
                        skip_res=self.input_shape[1:] / (2**count_downsample),
                        skip_channels=skip_channels_num,
                        gamma=gamma,
                    )
                )
                self.layer_to_block_map[current_layer_idx] = current_block_idx
                current_layer_idx += 1
            
            self._layers.append(
                XiConv(
                    int(num_filters[i + 1] * alpha),
                    int(num_filters[i + 1] * alpha),
                    kernel_size=3,
                    stride=1,
                    skip_tensor_in=True,
                    skip_res=self.input_shape[1:] / (2**count_downsample),
                    skip_channels=skip_channels_num,
                    gamma=gamma,
                )
            )
            self.layer_to_block_map[current_layer_idx] = current_block_idx
            current_layer_idx += 1
            current_block_idx += 1

        self.block_to_layer_map[current_block_idx] = current_layer_idx
        
        self._layers.append(
            XiConv(
                int(num_filters[-2] * alpha),
                int(num_filters[-1] * alpha),
                kernel_size=3,
                stride=1,
                skip_tensor_in=True,
                skip_res=self.input_shape[1:] / (2**count_downsample),
                skip_channels=skip_channels_num,
                attention=False,
            )
        )
        self.layer_to_block_map[current_layer_idx] = current_block_idx
        current_layer_idx += 1
        
        self._layers.append(
            XiConv(
                int(num_filters[-1] * alpha),
                int(num_filters[-1] * alpha),
                kernel_size=3,
                stride=1,
                skip_tensor_in=True,
                skip_res=self.input_shape[1:] / (2**count_downsample),
                skip_channels=skip_channels_num,
                attention=False,
            )
        )
        self.layer_to_block_map[current_layer_idx] = current_block_idx
        
        if self.return_blocks is not None:
            print(f"XiNet configured to return blocks {self.return_blocks}:")
            print(f"Block to layer mapping: {self.block_to_layer_map}")
            
            # Convert return_blocks to return_layers
            self.return_layers = []
            for block_idx in self.return_blocks:
                if block_idx in self.block_to_layer_map:
                    if block_idx < current_block_idx:
                        last_layer_of_block = self.block_to_layer_map[block_idx + 1] - 1
                    else:
                        last_layer_of_block = current_layer_idx
                    
                    self.return_layers.append(last_layer_of_block)
                    print(f"Block {block_idx} corresponds to layer {last_layer_of_block}")
                else:
                    print(f"Warning: Block {block_idx} not found in the network")
        else:
            self.return_layers = None

        self.input_shape = input_shape
        if self.include_top:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(int(num_filters[-1] * alpha), num_classes),
            )

    def forward(self, x):
        """Computes the forward step of the XiNet.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Output of the network, as defined from
            the init. : Union[torch.Tensor, Tuple]
        """
        x = self.conv1(x)
        skip = None
        ret = []
        for layer_id, layer in enumerate(self._layers):
            if layer_id == 0:
                x = layer(x)
                skip = x
            else:
                x = layer([x, skip])

            if self.return_layers is not None and layer_id in self.return_layers:
                ret.append(x)

        if self.include_top:
            x = self.classifier(x)

        if self.return_blocks is not None:
            return x, ret

        return x