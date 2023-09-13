"""
File: train.py
Project: project
Created Date: 2023-09-12 15:55:30
Author: chenkaixu
-----
Comment:

This is the train file for the project. 
It includes the train logic and val logic, under the pytorch-lightning framework.

Have a good code time!
-----
Last Modified: 2023-09-13 07:22:25
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-09-13	KX.C	if self.log have some promble, can use wandb.log to replace.
2023-09-13	KX.C	when calc the acc, if not drop last will occure err, tuple index out of range.

"""

import csv, logging
from typing import Any, List, Optional, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import LightningModule

from torchvision.io import write_video
from torchvision.utils import save_image, flow_to_image

from torchmetrics import classification
from torchmetrics.functional.classification import (
    binary_f1_score,
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_cohen_kappa,
    binary_auroc,
    binary_confusion_matrix,
)

from models.make_model import MakeVideoModule
from models.optical_flow import Optical_flow

import wandb


class TwoStreamLightningModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr

        self.num_classes = hparams.model.model_class_num

        self.optical_flow_model = Optical_flow()
        # define model
        self.video_cnn = MakeVideoModule(hparams).make_walk_resnet(3)
        self.of_cnn = MakeVideoModule(hparams).make_walk_resnet(2)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self._accuracy = classification.BinaryAccuracy()
        self._precision = classification.BinaryPrecision()
        self._recall = classification.BinaryRecall()
        self._confusion_matrix = classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        self.logger.watch(self.video_cnn)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float().squeeze()  # b

        b, c, t, h, w = video.shape

        video_flow = self.optical_flow_model.process_batch(video)  # b, c, t, h, w

        video_preds = self.video_cnn(video).squeeze()
        of_preds = self.of_cnn(video_flow).squeeze()

        video_loss = F.binary_cross_entropy_with_logits(video_preds, label)
        of_loss = F.binary_cross_entropy_with_logits(of_preds, label)

        total_loss = (video_loss + of_loss) / 2

        self.save_log([video_preds, of_preds], label, train_flag=True)

        # return [video, video_flow]
        return total_loss

    # TODO maybe need to store the first batch image and flow.
    # def training_epoch_end(self, outputs: list) -> None:

    #     images = wandb.Image(
    #         outputs[0][0][0].permute(1,0,2,3).cpu()*255,
    #     )
    #     flows = wandb.Image(
    #         outputs[0][1][0].permute(1,0,2,3).cpu()*255,
    #     )

    #     wandb.log({"train/image_batch0": images,
    #                "train/flow_batch0": flows})

    def on_validation_start(self) -> None:

        wandb.define_metric('val/loss', summary='min')
        wandb.define_metric('val/video_acc', summary='max')
        wandb.define_metric('val/of_acc', summary='max')
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float().squeeze()  # b

        b, c, t, h, w = video.shape

        video_flow = self.optical_flow_model.process_batch(video)  # b, c, t, h, w

        with torch.no_grad():
            video_preds = self.video_cnn(video).squeeze()
            of_preds = self.of_cnn(video_flow).squeeze()

        self.save_log([video_preds, of_preds], label, train_flag=False)

        # save one batch image and flow
        if batch_idx == 0:
            images = wandb.Image(
                video.resize(b*t, 3, h, w).cpu() * 255,
            )
            flows = wandb.Image(
                video_flow.resize(b*(t-1), 2, h, w).cpu() * 255,
            )

            wandb.log({f"Media/val_image_batch0_{label.tolist()}": images, f"Media/val_flow_batch0_{label.tolist()}": flows})

        # return video, video_flow  # video_preds_sigmoid, label

    def on_validation_epoch_end(self) -> None:
        self.log("lr", self.lr)

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min"),
                "monitor": "val/loss",
            },
        }

    def _get_name(self):
        return self.model_type

    def save_log(self, pred_list: list, label: torch.Tensor, train_flag: True):
        """
        save_log, save the log to the wandb.

        Args:
            pred_list (list): the predicted value in list, [video_preds, of_preds]
            label (torch.Tensor): ground truth label.
            train_flag (True): train flag, for the wandb log.
        """

        if train_flag:
            flag = "train"
        else:
            flag = "val"

        video_preds = pred_list[0]
        of_preds = pred_list[1]

        video_preds_sigmoid = torch.sigmoid(video_preds)
        of_preds_sigmoid = torch.sigmoid(of_preds)

        video_loss = F.binary_cross_entropy_with_logits(video_preds, label)
        of_loss = F.binary_cross_entropy_with_logits(of_preds, label)

        total_loss = (video_loss + of_loss) / 2
        self.log(f"{flag}/loss", total_loss)

        # video metrics
        video_acc = binary_accuracy(video_preds_sigmoid, label)
        video_precision = binary_precision(video_preds_sigmoid, label)
        video_recall = binary_recall(video_preds_sigmoid, label)
        video_confusion_matrix = binary_confusion_matrix(video_preds_sigmoid, label)

        self.log_dict(
            {
                f"{flag}/video_acc": video_acc,
                f"{flag}/video_precision": video_precision,
                f"{flag}/video_recall": video_recall,
            }
        )

        logging.info("*" * 50)
        logging.info(f"{flag}/video_confusion_matrix: %s" % video_confusion_matrix)

        # of metrics
        of_acc = binary_accuracy(of_preds_sigmoid, label)
        of_precision = binary_precision(of_preds_sigmoid, label)
        of_recall = binary_recall(of_preds_sigmoid, label)
        of_confusion_matrix = binary_confusion_matrix(of_preds_sigmoid, label)

        self.log_dict(
            {
                f"{flag}/of_acc": of_acc,
                f"{flag}/of_precision": of_precision,
                f"{flag}/of_recall": of_recall,
            }
        )

        logging.info("*" * 50)
        logging.info(f"{flag}/of_confusion_matrix: %s" % of_confusion_matrix)
