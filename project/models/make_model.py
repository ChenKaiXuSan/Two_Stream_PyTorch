# %%
from pytorchvideo.models import x3d, resnet, csn, slowfast, r2plus1d
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%

class MakeVideoModule(nn.Module):
    '''
    the module zoo from the PytorchVideo lib.

    Args:
        nn (_type_): 
    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class_num
        self.model_depth = hparams.model_depth

        self.transfor_learning = True

    def make_walk_csn(self):
        return csn.create_csn(
            input_channel=3,
            model_depth=self.model_depth,
            model_num_class=self.model_class_num,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

    def make_walk_resnet(self):

        if self.transfor_learning:
            slow = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            
            # change the knetics-400 output 400 to model class num
            slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
        else:
            slow = resnet.create_resnet(
                input_channel=3,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )

        return slow

    def make_walk_x3d(self) -> nn.Module:
        return x3d.create_x3d(
            input_channel=3,
            model_num_class=self.model_class_num,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
            input_clip_length=4,
        )

    # todo question with input tensor
    def make_walk_slowfast(self) -> nn.Module:
        return slowfast.create_slowfast(

        )

    # todo question with input tensor
    def make_walk_r2plus1d(self) -> nn.Module:
        return r2plus1d.create_r2plus1d(
            input_channel=3,
            model_num_class=self.model_class_num,
            model_depth=self.model_depth,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )
