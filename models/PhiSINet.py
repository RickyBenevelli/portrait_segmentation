import torch
import torch.nn as nn
import torch.nn.functional as F

from .phinet import PhiNet

BN_moment = 0.1

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class BR(nn.Module):
    '''
    This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, group=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
                              padding=(padding, padding), bias=False, groups=group)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class PhiSINet_Encoder(nn.Module):
    """
    Encoder part of the PhiSINet architecture, based on the PhiNet implementation.
    
    Arguments
    ---------
    classes: int
        Number of output classes.
    input_shape: list
        Shape of the input tensor [channels, height, width].
    alpha: float
        Width multiplier.
    beta: float
        Depth multiplier for PhiNet.
    t_zero: float
        Expansion factor in bottleneck blocks for PhiNet.
    num_layers: int
        Number of layers in the PhiNet backbone.
    skip_layer: int
        Index of the layer to extract features from for skip connection.
    """
    def __init__(
        self,
        classes: int = 20,
        input_shape: list = [3, 224, 224],
        alpha: float = 1.0,
        beta: float = 1.0,
        t_zero: float = 6.0,
        num_layers: int = 5,
        skip_layer: int = 3,
        divisor: int = 1
    ):
        super().__init__()
        self.skip_layer = skip_layer
        
        self.encoder = PhiNet(
            input_shape=input_shape,
            num_layers=num_layers,
            alpha=alpha,
            beta=beta,
            t_zero=t_zero,
            include_top=False,
            return_layers=[self.skip_layer],
            divisor=divisor
        )
        
        base_filters = 48
        b1_filters = 24
        b2_filters = 48
        
        if skip_layer <= 3:
            self.dim2 = _make_divisible(int(b1_filters * alpha), divisor=divisor)
        else:
            block_filters = b2_filters * (2 ** (skip_layer // 2 - 1))
            self.dim2 = _make_divisible(int(block_filters * alpha), divisor=divisor)
        
        downsampling_layers = [5, 7]  # Default from PhiNet
        block_filters = b2_filters
        
        doubles = 0
        for i in range(4, num_layers + 1):
            if i in downsampling_layers:
                block_filters *= 2
                doubles += 1
        
        self.dim3 = _make_divisible(int(block_filters * alpha), divisor=divisor)
        
        self.classifier = nn.Conv2d(self.dim3, classes, 1, bias=False)
        self.classes = classes
        
        print(f"Initialized classifier with input channels: {self.dim3}, output channels: {self.classes}")
        
        self.skip_layer_tensor = None

    def forward(self, x):
        output, ret_blocks = self.encoder(x)
        
        self.skip_layer_tensor = ret_blocks[0]
        classifier_output = self.classifier(output)
        # print(f"DEBUG: output: {output.shape}, classifier_output: {classifier_output.shape}")
        # output: torch.Size([36, 96, 14, 14]), classifier_output: torch.Size([36, 1, 14, 14])
        
        return classifier_output


class SINet_PhiNet(nn.Module):
    """
    SINet architecture using the PhiNet as the backbone.
    
    Arguments
    ---------
    classes: int
        Number of output classes.
    input_shape: list
        Shape of the input tensor [channels, height, width].
    alpha: float
        Width multiplier.
    beta: float
        Depth multiplier for PhiNet.
    t_zero: float
        Expansion factor in bottleneck blocks for PhiNet.
    num_layers: int
        Number of layers in the PhiNet backbone.
    skip_layer: int
        Index of the layer to extract features from for skip connection.
    encoderFile: str
        Path to pretrained encoder weights file.
    """
    def __init__(
        self,
        classes: int = 20,
        input_shape: list = [3, 224, 224],
        alpha: float = 1.0,
        beta: float = 1.0,
        t_zero: float = 6.0,
        num_layers: int = 5,
        skip_layer: int = 3,
        encoderFile: str = None,
        divisor: int = 1
    ):
        super().__init__()
        
        self.encoder = PhiSINet_Encoder(
            classes=classes,
            input_shape=input_shape,
            alpha=alpha,
            beta=beta,
            t_zero=t_zero,
            num_layers=num_layers,
            skip_layer=skip_layer,
            divisor=divisor
        )
        
        if encoderFile is not None:
            if torch.cuda.device_count() == 0:
                self.encoder.load_state_dict(torch.load(encoderFile, map_location="cpu"))
            else:
                self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')
        
        base_filters = 48
        b1_filters = 24
        b2_filters = 48
        
        if skip_layer <= 3:
            self.skip_dim = _make_divisible(int(b1_filters * alpha), divisor=divisor)
        else:
            block_filters = b2_filters * (2 ** (skip_layer // 2 - 1))
            self.skip_dim = _make_divisible(int(block_filters * alpha), divisor=divisor)
        
        # Calculate encoder output dimensions
        downsampling_layers = [5, 7]  # Default from PhiNet
        block_filters = b2_filters
        
        doubles = 0
        for i in range(4, num_layers + 1):
            if i in downsampling_layers:
                block_filters *= 2
                doubles += 1
        
        self.enc_dim = _make_divisible(int(block_filters * alpha), divisor=divisor)
        self.classes = classes
        
        # Decode components
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn_3 = nn.BatchNorm2d(classes, eps=1e-3)
        
        # Initialize the skip connection processing
        self.level2_C = nn.Sequential(
            nn.Conv2d(self.skip_dim, self.classes, 1, bias=False),
            nn.BatchNorm2d(self.classes, eps=1e-3),
            nn.PReLU(self.classes)
        )
        
        self.bn_2 = nn.BatchNorm2d(classes, eps=1e-3)
        
        self.classifier = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(classes, classes, 3, padding=1, bias=False)
        )

    def forward(self, input):
        Enc_final = self.encoder(input)
        skip_layer_tensor = self.encoder.skip_layer_tensor
        
        # First decoder stage - upsample encoder output
        Dnc_stage1 = self.bn_3(self.up(Enc_final))

        # Calculate confidence map for gating
        confidence = torch.max(F.softmax(Dnc_stage1, dim=1), dim=1)[0]
        b, c, h, w = Dnc_stage1.shape
        stage1_gate = (1 - confidence).unsqueeze(1).expand(b, c, h, w)
        
        # Second decoder stage with gated skip connection
        Dnc_stage2_0 = self.level2_C(skip_layer_tensor)
        Dnc_stage2 = self.bn_2(self.up(Dnc_stage2_0 * stage1_gate + Dnc_stage1))
        
        # Final classification
        classifier = self.classifier(Dnc_stage2)
        
        return classifier



def _make_divisible(v, divisor=8, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by divisor.
    
    Arguments
    ---------
    v : int
        The original number of channels.
    divisor : int, optional
        The divisor to ensure divisibility (default is 8).
    min_value : int or None, optional
        The minimum value for the divisible channels (default is None).
        
    Returns
    -------
    int
        The adjusted number of channels.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def Enc_PhiSINet(
    classes: int = 20,
    input_shape: list = [3, 224, 224],
    alpha: float = 1.0,
    beta: float = 1.0,
    t_zero: float = 6.0,
    num_layers: int = 5,
    skip_layer: int = 3,
) -> PhiSINet_Encoder:
    """
    Factory function to create a PhiSINet encoder.
    """
    return PhiSINet_Encoder(
        classes=classes, 
        input_shape=input_shape, 
        alpha=alpha, 
        beta=beta,
        t_zero=t_zero,
        num_layers=num_layers,
        skip_layer=skip_layer,
    )


def Dnc_PhiSINet(
    classes: int = 20,
    input_shape: list = [3, 224, 224],
    alpha: float = 1.0,
    beta: float = 1.0,
    t_zero: float = 6.0,
    num_layers: int = 5,
    skip_layer: int = 3,
    encoderFile: str = None
) -> SINet_PhiNet:
    """
    Factory function to create a complete SINet with PhiNet backbone.
    """
    return SINet_PhiNet(
        classes=classes, 
        input_shape=input_shape, 
        alpha=alpha, 
        beta=beta,
        t_zero=t_zero,
        num_layers=num_layers,
        skip_layer=skip_layer,
        encoderFile=encoderFile
    )