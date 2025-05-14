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
    def __init__(
        self,
        classes: int = 20,
        input_shape: list = [3, 224, 224],
        alpha: float = 1.0,
        gamma: float = 4.0,
        num_layers: int = 5,
        encoderFile: str = None
    ):
        super().__init__()
        
        self.mid_block = 0 # num_layers // 2
        self.mid_layer_id = 0 # self.mid_block * 2
        # last_layer_id = 2 * num_layers - 3
        self.encoder_net = XiNet(
            input_shape=input_shape,
            alpha=alpha,
            gamma=gamma,
            num_layers=num_layers,
            include_top=False,
            return_layers=[self.mid_layer_id]
        )
        if encoderFile is not None:
            loc = "cpu" if torch.cuda.device_count() == 0 else None
            self.encoder_net.load_state_dict(torch.load(encoderFile, map_location=loc))

        base_filters = 16
        num_filters = [int(2 ** (base_filters**0.5 + i)) for i in range(num_layers)]
        self.dim2 = int(num_filters[self.mid_block+1] * alpha)
        self.dim3 = int(num_filters[-1] * alpha)

        # testa di classificazione a 1/8 di risoluzione
        self.classifier = nn.Conv2d(self.dim3, classes, 1, bias=False)
        
        self.mid_layer = None

    def forward(self, x):
        # ritorna l'output e il layer a metà: risultato (1/8) e mid-level (1/4)
        output, ret_layers = self.encoder_net(x)
        
        self.mid_layer = ret_layers[0]
        
        # Add upsampling to match SINet's output resolution (from 14×14 to 28×28)
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
        
        classifier_output = self.classifier(output) 
        # output = torch.Size([36, 256, 28, 28]) 
        # classifier_output = torch.Size([36, 1, 28, 28])
        
        return classifier_output


class SINet_XiNet(nn.Module):
    def __init__(
        self,
        classes: int = 20,
        input_shape: list = [3, 224, 224],
        alpha: float = 1.0,
        gamma: float = 4.0,
        num_layers: int = 5,
        encoderFile: str = None
    ):
        super().__init__()

        self.encoder = XiSINet_Encoder(
            classes=classes,
            input_shape=input_shape,
            alpha=alpha,
            gamma=gamma,
            num_layers=num_layers,
            encoderFile=encoderFile
        )

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
        mid_layer = self.encoder.mid_layer
        
        Dnc_stage1 = self.bn_3(self.up(Enc_final))
        
        confidence = torch.max(F.softmax(Dnc_stage1, dim=1), dim=1)[0]
        b, c, h, w = Dnc_stage1.shape
        stage1_gate = (1 - confidence).unsqueeze(1).expand(b, c, h, w)
        
        Dnc_stage2_0 = self.level2_C(mid_layer)
        Dnc_stage2 = self.bn_2(self.up(Dnc_stage2_0 * stage1_gate + Dnc_stage1))
        
        classifier = self.classifier(Dnc_stage2)
        # mid_layer: torch.Size([24, 32, 56, 56])
        # Dnc_stage2_0: torch.Size([24, 1, 56, 56])
        # stage1_gate: torch.Size([24, 1, 56, 56])
        # Dnc_stage1: torch.Size([24, 1, 56, 56])
        # Dnc_stage2: torch.Size([24, 1, 112, 112])
        # classifier: torch.Size([24, 1, 224, 224])
        
        return classifier


def Enc_XiSINet(
    classes: int = 20,
    input_shape: list = [3,224,224],
    alpha: float = 1.0,
    gamma: float = 4.0,
    num_layers: int = 5,
    encoderFile: str = None
) -> XiSINet_Encoder:
    return XiSINet_Encoder(classes, input_shape, alpha, gamma, num_layers, encoderFile)

def Dnc_XiSINet(
    classes: int = 20,
    input_shape: list = [3,224,224],
    alpha: float = 1.0,
    gamma: float = 4.0,
    num_layers: int = 5,
    encoderFile: str = None
) -> SINet_XiNet:
    return SINet_XiNet(classes, input_shape, alpha, gamma, num_layers, encoderFile)