import torch
import torch.nn as nn
import torch.nn.functional as F

#from micromind.networks import XiNet
from .xinet import XiNet

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


class XiSINet_Encoder(nn.Module):
    """
    Encoder part of the XiSINet architecture, based on the modified XiNet implementation.
    
    Arguments
    ---------
    classes: int
        Number of output classes.
    input_shape: list
        Shape of the input tensor [channels, height, width].
    alpha: float
        Width multiplier.
    gamma: float
        Compression factor for XiNet.
    num_layers: int
        Number of layers in the XiNet backbone.
    min_feature_size: int
        Minimum feature size before stopping downsampling.
    skip_layer: int
        Index of the middle block to extract features from.
    """
    def __init__(
        self,
        classes: int = 20,
        input_shape: list = [3, 224, 224],
        alpha: float = 1.0,
        gamma: float = 4.0,
        num_layers: int = 5,
        min_feature_size: int = 7,
        skip_layer: int = 0
    ):
        super().__init__()
        self.skip_layer = skip_layer
        
        self.encoder = XiNet(
            input_shape=input_shape,
            alpha=alpha,
            gamma=gamma,
            num_layers=num_layers,
            include_top=False,
            min_feature_size=min_feature_size,
            return_blocks=[self.skip_layer]
        )

        base_filters = 16
        num_filters = [int(2 ** (base_filters**0.5 + i)) for i in range(num_layers)]
        self.dim2 = int(num_filters[self.skip_layer+1] * alpha)
        self.dim3 = int(num_filters[-1] * alpha)

        self.classifier = nn.Conv2d(self.dim3, classes, 1, bias=False)
        
        self.skip_layer_tensor = None

    def forward(self, x):
        output, ret_blocks = self.encoder(x)
        
        self.skip_layer_tensor = ret_blocks[0]
        
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)        
        classifier_output = self.classifier(output)
        #print(f"DEBUG: output: {output.shape}, classifier_output: {classifier_output.shape}")
        # output = torch.Size([36, 128, 28, 28], classifier_output = torch.Size([36, 1, 28, 28]
        return classifier_output


class SINet_XiNet(nn.Module):
    """
    SINet architecture using the modified XiNet as the backbone.
    
    Arguments
    ---------
    classes: int
        Number of output classes.
    input_shape: list
        Shape of the input tensor [channels, height, width].
    alpha: float
        Width multiplier.
    gamma: float
        Compression factor for XiNet.
    num_layers: int
        Number of layers in the XiNet backbone.
    min_feature_size: int
        Minimum feature size before stopping downsampling.
    skip_layer: int
        Index of the middle block to extract features from.
    encoderFile: str
        Path to pretrained encoder weights file.
    """
    def __init__(
        self,
        classes: int = 20,
        input_shape: list = [3, 224, 224],
        alpha: float = 1.0,
        gamma: float = 4.0,
        num_layers: int = 5,
        min_feature_size: int = 7,
        skip_layer: int = 0,
        encoderFile: str = None
    ):
        super().__init__()

        self.encoder = XiSINet_Encoder(
            classes=classes,
            input_shape=input_shape,
            alpha=alpha,
            gamma=gamma,
            num_layers=num_layers,
            min_feature_size=min_feature_size,
            skip_layer=skip_layer
        )
        
        if encoderFile is not None:
            if torch.cuda.device_count() == 0:
                self.encoder.load_state_dict(torch.load(encoderFile, map_location="cpu"))
            else:
                self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn_3 = nn.BatchNorm2d(classes, eps=1e-3)
        
        self.level2_C = nn.Sequential(
            nn.Conv2d(self.encoder.dim2, classes, 1, bias=False),
            nn.BatchNorm2d(classes, eps=1e-3),
            nn.PReLU(classes)
        )
        
        self.bn_2 = nn.BatchNorm2d(classes, eps=1e-3)
        
        self.classifier = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(classes, classes, 3, padding=1, bias=False)
        )

    def forward(self, input):
        Enc_final = self.encoder(input)
        skip_layer_tensor = self.encoder.skip_layer_tensor
        
        # First decoder stage
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


def Enc_XiSINet(
    classes: int = 20,
    input_shape: list = [3, 224, 224],
    alpha: float = 1.0,
    gamma: float = 4.0,
    num_layers: int = 5,
    min_feature_size: int = 7,
    skip_layer: int = 0
) -> XiSINet_Encoder:
    """
    Factory function to create an XiSINet encoder.
    """
    return XiSINet_Encoder(
        classes=classes, 
        input_shape=input_shape, 
        alpha=alpha, 
        gamma=gamma, 
        num_layers=num_layers,
        min_feature_size=min_feature_size,
        skip_layer=skip_layer
    )


def Dnc_XiSINet(
    classes: int = 20,
    input_shape: list = [3, 224, 224],
    alpha: float = 1.0,
    gamma: float = 4.0,
    num_layers: int = 5,
    min_feature_size: int = 7,
    skip_layer: int = 0,
    encoderFile: str = None
) -> SINet_XiNet:
    """
    Factory function to create a complete SINet with XiNet backbone.
    """
    return SINet_XiNet(
        classes=classes, 
        input_shape=input_shape, 
        alpha=alpha, 
        gamma=gamma, 
        num_layers=num_layers,
        min_feature_size=min_feature_size,
        skip_layer=skip_layer,
        encoderFile=encoderFile
    )