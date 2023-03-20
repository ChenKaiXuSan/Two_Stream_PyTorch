# %%
from pytorchvideo.models import x3d, resnet, csn, slowfast, r2plus1d

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# %%

class MakeVideoModule(nn.Module):
    '''
    the module zoo from the PytorchVideo lib, to make the different 3D model.

    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class_num
        self.model_depth = hparams.model_depth

        self.transfor_learning = hparams.transfor_learning

    def make_walk_resnet(self, input_channel:int = 3) -> nn.Module:

        if self.transfor_learning:
            slow = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            
            # for the folw model and rgb model 
            slow.blocks[0].conv = nn.Conv3d(input_channel, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
            # change the knetics-400 output 400 to model class num
            slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        else:
            slow = resnet.create_resnet(
                input_channel=input_channel,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )

        return slow

    def make_walk_x3d(self, input_channel:int = 3) -> nn.Module:

        if self.transfor_learning:
            # x3d l model, param 6.15 with 16 frames. more smaller maybe more faster.
            # top1 acc is 77.44
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='x3d_m', pretrained=True)
            model.blocks[0].conv.conv_t = nn.Conv3d(
                in_channels=input_channel, 
                out_channels=24,
                kernel_size=(1,3,3),
                stride=(1,2,2),
                padding=(0,1,1),
                bias=False
                )
            
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
            model.blocks[-1].activation = None

        else:
            model = x3d.create_x3d(
                input_channel=3,
                input_clip_length=16,
                input_crop_size=224,
                model_num_class=1,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                head_activation=None,
            )

        return model
    
class MakeOriginalTwoStream(nn.Module):
    '''
    from torchvision make resnet 50 network.
    input is single figure.
    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class_num
        self.transfor_learning = hparams.transfor_learning

    def make_resnet(self, input_channel:int = 3):

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        # from pytorchvision, use resnet 50.
        # weights = ResNet50_Weights.DEFAULT
        # model = resnet50(weights=weights)

        # for the folw model and rgb model 
        model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # change the output 400 to model class num
        model.fc = nn.Linear(2048, self.model_class_num)

        return model