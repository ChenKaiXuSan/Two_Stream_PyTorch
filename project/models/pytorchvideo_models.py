# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.io import write_video

from models.make_model import MakeVideoModule
from models.optical_flow import Optical_flow
from models.instance_segmentation import batch_instance_segmentation

from pytorch_lightning import LightningModule

from utils.metrics import *

# %%

class WalkVideoClassificationLightningModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type = hparams.model
        self.img_size = hparams.img_size

        self.lr = hparams.lr
        self.num_class = hparams.model_class_num

        # model define
        self.optical_flow_model = Optical_flow()
        self.instance_segmentation = batch_instance_segmentation()
        self.model = MakeVideoModule(hparams)

        # select the network structure
        if self.model_type == 'resnet':
            self.model_rgb = self.model.make_walk_resnet(3)
            self.model_flow = self.model.make_walk_resnet(2)

        elif self.model_type == 'csn':
            self.model = self.model.make_walk_csn()

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        # select the metrics
        self._accuracy = get_Accuracy(self.num_class)
        self._precision = get_Precision(self.num_class)
        self._confusion_matrix = get_Confusion_Matrix()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        train steop when trainer.fit called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
        '''

        label = batch['label'].detach()  # b, c, t, h, w
        video = batch['video'].detach()  # b, c, t, h, w

        self.model_rgb.train()
        self.model_flow.train()

        # add instance segmentation mask to raw video
        masked_video = self.instance_segmentation.handle_batch_video(video) # b, c, t, h, w

        # pred the optical flow
        video_flow = self.optical_flow_model.process_batch(masked_video)  # b, c, t, h, w
        video_rgb = masked_video[:, :, :-1, :, :]  # dont use the last frame imge

        write_video('/workspace/Two_Stream_PyTorch/tests/test_flow.mp4', video_rgb[0].permute(1, 2, 3, 0).cpu(), fps=30)

        # classification task
        pred_video_rgb = self.model_rgb(video_rgb)
        # pred_video_flow = self.model_flow(video_flow)
        
        pred_video_rgb_sigmoid = torch.sigmoid(pred_video_rgb).squeeze(dim=-1)
        # pred_video_flow_sigmoid = torch.sigmoid(pred_video_flow).squeeze(dim=-1)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        loss_rgb = F.binary_cross_entropy_with_logits(pred_video_rgb.squeeze(dim=-1), label.float())
        # loss_flow = F.binary_cross_entropy_with_logits(pred_video_flow.squeeze(dim=-1), label.float())

        loss = loss_rgb #+ loss_flow

        # soft_margin_loss = F.soft_margin_loss(pred_video_sigmoid, label.float())

        accuracy_rgb = self._accuracy(pred_video_rgb_sigmoid, label)
        # accuracy_flow = self._accuracy(pred_video_flow_sigmoid, label)

        self.log('train_loss', loss)
        self.log('train_acc', accuracy_rgb)

        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     '''
    #     after validattion_step end.

    #     Args:
    #         outputs (list): a list of the train_step return value.
    #     '''

    #     # log epoch metric
    #     # self.log('train_acc_epoch', self.accuracy)
    #     pass

    def validation_step(self, batch, batch_idx):
        '''
        val step when trainer.fit called.

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss 
            accuract: selected accuracy result.
        '''

        # input and model define
        label = batch['label'].detach()  # b, c, t, h, w
        video = batch['video'].detach()  # b, c, t, h, w

        self.model_rgb.eval()
        self.model_flow.eval()

        # add instance segmentation mask to raw video
        masked_video = self.instance_segmentation.handle_batch_video(video) # b, c, t, h, w
        
        # pred the optical flow base RAFT
        # video_flow = self.optical_flow_model.process_batch(masked_video)  # b, c, t, h, w
        video_rgb = masked_video[:, :, :-1, :, :]  # dont use the last frame imge

        # classification task
        with torch.no_grad():
            pred_video_rgb = self.model_rgb(video_rgb)
            # pred_video_flow = self.model_flow(video_flow)

        pred_video_rgb_sigmoid = torch.sigmoid(pred_video_rgb).squeeze(dim=-1)
        # pred_video_flow_sigmoid = torch.sigmoid(pred_video_flow).squeeze(dim=-1)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        loss_rgb = F.binary_cross_entropy_with_logits(pred_video_rgb.squeeze(dim=-1), label.float())
        # loss_flow = F.binary_cross_entropy_with_logits(pred_video_flow.squeeze(dim=-1), label.float())

        loss = loss_rgb #+ loss_flow

        # soft_margin_loss = F.soft_margin_loss(pred_video_sigmoid, label.float())

        # calc the metric, function from torchmetrics
        accuracy_rgb = self._accuracy(pred_video_rgb_sigmoid, label)
        # accuracy_flow = self._accuracy(pred_video_flow_sigmoid, label)

        precision_rgb = self._precision(pred_video_rgb_sigmoid, label)
        # precision_flow = self._precision(pred_video_flow_sigmoid, label)

        confusion_matrix = self._confusion_matrix(pred_video_rgb_sigmoid, label)

        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'val_loss': loss, 'val_acc_rgb': accuracy_rgb, 'val_precision_rgb': precision_rgb,
                       #'val_acc_flow': accuracy_flow, 'val_precision_flow': precision_flow
                       }, on_step=False, on_epoch=True)

        return accuracy_rgb

    # def validation_epoch_end(self, outputs):

    #     val_metric = torch.stack(outputs, dim=0)

    #     final_acc = torch.sum(val_metric) / len(val_metric)

    #     print(final_acc)

    #     return final_acc

    def test_step(self, batch, batch_idx):
        '''
        test step when trainer.test called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_
        '''

        # input and model define
        label = batch['label'].detach()  # b
        video = batch['video'].detach()  # b, c, t, h, w

        self.model_rgb.eval()
        self.model_flow.eval()

        # pred the optical flow base RAFT
        video_flow = self.optical_flow_model.process_batch(video)  # b, c, t, h, w
        video_rgb = video[:, :, :-1, :, :]  # dont use the last frame image (b, c, t, h, w)

        # eval model, feed data here
        with torch.no_grad():
            pred_video_rgb = self.model_rgb(video_rgb)
            pred_video_flow = self.model_flow(video_flow)

        pred_video_rgb_sigmoid = torch.sigmoid(pred_video_rgb).squeeze(dim=-1)
        pred_video_flow_sigmoid = torch.sigmoid(pred_video_flow).squeeze(dim=-1)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        loss_rgb = F.binary_cross_entropy_with_logits(pred_video_rgb.squeeze(dim=-1), label.float())
        loss_flow = F.binary_cross_entropy_with_logits(pred_video_flow.squeeze(dim=-1), label.float())

        loss = loss_rgb + loss_flow

        # calc the metric, function from torchmetrics
        accuracy_rgb = self._accuracy(pred_video_rgb_sigmoid, label)
        accuracy_flow = self._accuracy(pred_video_flow_sigmoid, label)

        precision_rgb = self._precision(pred_video_rgb_sigmoid, label)
        precision_flow = self._precision(pred_video_flow_sigmoid, label)

        confusion_matrix = self._confusion_matrix(pred_video_rgb_sigmoid, label)

        # log the test loss, and test acc, in step and in epoch
        self.log_dict({'test_loss': loss, 'test_acc_rgb': accuracy_rgb, 'test_acc_flow': accuracy_flow,
                      'test_precision_rgb': precision_rgb, 'test_precision_flow': precision_flow}, on_step=False, on_epoch=True)

        return accuracy_rgb, accuracy_flow

    def test_epoch_end(self, outputs):

        # test_metric = torch.stack(outputs, dim=0)

        # final_acc = torch.sum(test_metric) / len(test_metric)

        # print(final_acc)
        pass

    def configure_optimizers(self):
        '''
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        '''

        optimzier = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimzier,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimzier),
                "monitor": "val_loss",
            },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type
