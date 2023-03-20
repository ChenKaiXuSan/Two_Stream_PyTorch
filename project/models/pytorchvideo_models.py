# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.io import write_video

from models.make_model import MakeVideoModule, MakeOriginalTwoStream
from models.optical_flow import Optical_flow

from pytorch_lightning import LightningModule

from torchmetrics.functional.classification import \
    binary_f1_score, \
    binary_accuracy, \
    binary_cohen_kappa, \
    binary_auroc, \
    binary_confusion_matrix

# %%

class WalkVideoClassificationLightningModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type = hparams.model
        self.img_size = hparams.img_size

        self.lr = hparams.lr

        # model define
        self.optical_flow_model = Optical_flow()

        if self.model_type == 'multi':
            self.model = MakeVideoModule(hparams)
            self.model_rgb = self.model.make_walk_x3d(3)
            self.model_flow = self.model.make_walk_x3d(2)
        else:
            self.model = MakeOriginalTwoStream(hparams)
            self.model_rgb = self.model.make_resnet(3)
            self.model_flow = self.model.make_resnet(2)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

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

        if self.model_type == 'multi':
            pred_video_rgb, pred_video_rgb_sigmoid, pred_video_flow, pred_video_flow_sigmoid, loss = self.multi_logic(label, video)
        else:
            pred_video_rgb, pred_video_rgb_sigmoid, pred_video_flow, pred_video_flow_sigmoid, loss = self.multi_logic(label, video)

        accuracy_rgb = binary_accuracy(pred_video_rgb_sigmoid, label)

        # video rgb metrics
        accuracy = binary_accuracy(pred_video_rgb_sigmoid, label)
        f1_score = binary_f1_score(pred_video_rgb_sigmoid, label)
        auroc = binary_auroc(pred_video_rgb_sigmoid, label)
        cohen_kappa = binary_cohen_kappa(pred_video_rgb_sigmoid, label)
        cm = binary_confusion_matrix(pred_video_rgb_sigmoid, label)

        # log to tensorboard
        self.log_dict({'train_loss': loss,
                       'train_rgb_acc': accuracy,
                       'train_rgb_f1_score': f1_score, 
                       'train_rgb_auroc': auroc, 
                       'train_rgb_cohen_kappa': cohen_kappa, 
                       })
        
        # video optical metrics
        accuracy = binary_accuracy(pred_video_flow_sigmoid, label)
        f1_score = binary_f1_score(pred_video_flow_sigmoid, label)
        auroc = binary_auroc(pred_video_flow_sigmoid, label)
        cohen_kappa = binary_cohen_kappa(pred_video_flow_sigmoid, label)
        cm = binary_confusion_matrix(pred_video_flow_sigmoid, label)

        # log to tensorboard
        self.log_dict({'train_loss': loss,
                       'train_flow_acc': accuracy,
                       'train_flow_f1_score': f1_score, 
                       'train_flow_auroc': auroc, 
                       'train_flow_cohen_kappa': cohen_kappa, 
                       })


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
        label = batch['label'].detach()  # b
        video = batch['video'].detach()  # b, c, t, h, w

        if self.model_type == 'multi':
            pred_video_rgb, pred_video_rgb_sigmoid, pred_video_flow, pred_video_flow_sigmoid, loss = self.multi_logic(label, video)
        else:
            pred_video_rgb, pred_video_rgb_sigmoid, pred_video_flow, pred_video_flow_sigmoid, loss = self.single_logic(label, video)

        # video rgb metrics
        accuracy = binary_accuracy(pred_video_rgb_sigmoid, label)
        f1_score = binary_f1_score(pred_video_rgb_sigmoid, label)
        auroc = binary_auroc(pred_video_rgb_sigmoid, label)
        cohen_kappa = binary_cohen_kappa(pred_video_rgb_sigmoid, label)
        cm = binary_confusion_matrix(pred_video_rgb_sigmoid, label)

        # log to tensorboard
        self.log_dict({'val_loss': loss,
                       'val_rgb_acc': accuracy,
                       'val_rgb_f1_score': f1_score, 
                       'val_rgb_auroc': auroc, 
                       'val_rgb_cohen_kappa': cohen_kappa, 
                       })
        
        # video optical metrics
        accuracy = binary_accuracy(pred_video_flow_sigmoid, label)
        f1_score = binary_f1_score(pred_video_flow_sigmoid, label)
        auroc = binary_auroc(pred_video_flow_sigmoid, label)
        cohen_kappa = binary_cohen_kappa(pred_video_flow_sigmoid, label)
        cm = binary_confusion_matrix(pred_video_flow_sigmoid, label)

        # log to tensorboard
        self.log_dict({'val_loss': loss,
                       'val_flow_acc': accuracy,
                       'val_flow_f1_score': f1_score, 
                       'val_flow_auroc': auroc, 
                       'val_flow_cohen_kappa': cohen_kappa, 
                       })

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
        pass

    def test_epoch_end(self, outputs):
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

    def single_logic(self, label, video):
        pass

    def multi_logic(self, label, video):

        # pred the optical flow base RAFT
        last_frame = video[:, :, -1, :].unsqueeze(dim=2) # b, c, 1, h, w
        OF_video = torch.cat([video, last_frame], dim=2)
        video_flow = self.optical_flow_model.process_batch(OF_video)  # b, c, t, h, w

        # eval model, feed data here
        if self.training:
            pred_video_rgb = self.model_rgb(video)
            pred_video_flow = self.model_flow(video_flow)
        else:
            with torch.no_grad():
                pred_video_rgb = self.model_rgb(video)
                pred_video_flow = self.model_flow(video_flow)

        pred_video_rgb_sigmoid = torch.sigmoid(pred_video_rgb).squeeze(dim=-1)
        pred_video_flow_sigmoid = torch.sigmoid(pred_video_flow).squeeze(dim=-1)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        loss_rgb = F.binary_cross_entropy_with_logits(pred_video_rgb.squeeze(dim=-1), label.float())
        loss_flow = F.binary_cross_entropy_with_logits(pred_video_flow.squeeze(dim=-1), label.float())

        loss = loss_rgb + loss_flow

        return pred_video_rgb, pred_video_rgb_sigmoid, pred_video_flow, pred_video_flow_sigmoid, loss