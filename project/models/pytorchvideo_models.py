# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.make_model import MakeVideoModule

from pytorch_lightning import LightningModule

from utils.metrics import *

# %%

class WalkVideoClassificationLightningModule(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type=hparams.model
        self.img_size = hparams.img_size

        self.lr=hparams.lr
        self.num_class = hparams.model_class_num

        self.model = MakeVideoModule(hparams)

        # select the network structure 
        if self.model_type == 'resnet':
            self.model=self.model.make_walk_resnet()

        elif self.model_type == 'csn':
            self.model=self.model.make_walk_csn()

        elif self.model_type == 'x3d':
            self.model = self.model.make_walk_x3d()

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        # select the metrics
        self._accuracy = get_Accuracy(self.num_class)
        self._precision = get_Precision(self.num_class)
        self._confusion_matrix = get_Confusion_Matrix()

        # self.dice = get_Dice()
        # self.average_precision = get_Average_precision(self.num_class)
        # self.AUC = get_AUC()
        # self.f1_score = get_F1Score()
        # self.precision_recall = get_Precision_Recall()
        
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

        label = batch['label'].detach()

        # classification task
        y_hat = self.model(batch["video"])

        y_hat_sigmoid = torch.sigmoid(y_hat).squeeze(dim=-1)


        # squeeze(dim=-1) to keep the torch.Size([1]), not null.
        loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(dim=-1), label.float())
        soft_margin_loss = F.soft_margin_loss(y_hat_sigmoid, label.float())

        accuracy = self._accuracy(y_hat_sigmoid, label)

        self.log('train_loss', loss)
        self.log('train_acc', accuracy)

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

        label = batch['label'].detach()

        preds = self.model(batch["video"])

        preds_sigmoid = torch.sigmoid(preds).squeeze(dim=-1)

        # val_loss=F.cross_entropy(preds, label)
        val_loss = F.binary_cross_entropy_with_logits(preds.squeeze(dim=-1), label.float())

        # calc the metric, function from torchmetrics
        accuracy = self._accuracy(preds_sigmoid, label)

        precision = self._precision(preds_sigmoid, label)

        confusion_matrix = self._confusion_matrix(preds_sigmoid, label)

        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'val_loss': val_loss, 'val_acc': accuracy, 'val_average_precision': precision}, on_step=False, on_epoch=True)
        
        return accuracy

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

        labels = batch['label'].detach()

        preds = self.model(batch["video"])

        preds_softmax = torch.softmax(preds, dim=-1)
        preds_sigmoid = torch.sigmoid(preds).squeeze()

        # test_loss = F.cross_entropy(preds, labels)
        test_loss = F.binary_cross_entropy_with_logits(preds.squeeze(), labels.float())

        # calculate acc 
        accuracy = self._accuracy(preds_sigmoid, labels)

        average_precision = self._precision(preds_sigmoid, labels)
        # AUC = self.AUC(F.softmax(test_pred, dim=-1), batch["label"])
        # f1_score = self.f1_score(preds_softmax, labels)
        # precision, recall, threshold = self.precision_recall(F.softmax(test_pred, dim=-1), batch["label"])

        # self.dice(test_pred, target)

        # log the test loss, and test acc, in step and in epoch
        self.log_dict({'test_loss': test_loss, 'test_acc': accuracy, 'test_average_precision': average_precision}, on_step=False, on_epoch=True)

        return accuracy
        
    def test_epoch_end(self, outputs):

        test_metric = torch.stack(outputs, dim=0)

        final_acc = torch.sum(test_metric) / len(test_metric)
        
        print(final_acc)

    def configure_optimizers(self):
        '''
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        '''

        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type