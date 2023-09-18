'''
File: main.py
Project: project
Created Date: 2023-03-20 16:13:02
Author: chenkaixu
-----
Comment:
This project were based the pytorch, pytorch lightning and pytorch video library, 
for rapid development.
 
Have a good code time!
-----
Last Modified: 2023-09-15 12:54:35
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''

import os, logging, time, sys
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# callbacks
from pytorch_lightning.callbacks import TQDMProgressBar, RichModelSummary, ModelCheckpoint, EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback, TrainingDataMonitor

from dataloader.data_loader import WalkDataModule
from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from train import TwoStreamLightningModule

import pytorch_lightning
import wandb
import hydra

# login the wandb
wandb.login(anonymous="allow", key="eeece7dd9910c3cc2be6ae3e2f8b9b666f878066")

def train(hparams):

    # fixme will occure bug, with deterministic = true
    seed_everything(42, workers=True)

    classification_module = TwoStreamLightningModule(hparams)

    # instance the data module
    data_module = WalkDataModule(hparams)

    # for the tensorboard
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(hparams.log_path, hparams.model), name=hparams.log_version, version=hparams.fold)

    # init wandb logger
    wandb_logger = WandbLogger(name='_'.join([hparams.train.version, hparams.model.model, hparams.train.fold]), 
                            project='two_stream_3DCNN',
                            save_dir=hparams.train.log_path,
                            version=hparams.train.fold,
                            log_model="all")

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val/loss:.2f}-{val/acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/acc",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=5,
        mode='min',
    )

    # bolts callbacks
    table_metrics_callback = PrintTableMetricsCallback()
    monitor = TrainingDataMonitor(log_every_n_steps=50)

    trainer = Trainer(
                      devices=[hparams.train.gpu_num,],
                      accelerator="gpu",
                      max_epochs=hparams.train.max_epochs,
                      logger= wandb_logger, # tb_logger
                      #   log_every_n_steps=100,
                      check_val_every_n_epoch=1,
                      callbacks=[progress_bar, rich_model_summary, table_metrics_callback, monitor, model_check_point, early_stopping],
                      #   deterministic=True
                      )

    # from the params
    # trainer = Trainer.from_argparse_args(hparams)

    # training and val
    trainer.fit(classification_module, data_module)

    # TODO take code the record the val figure, in wandb and local file.
    trainer.validate(classification_module, data_module) # , ckpt_path='best')

    # when one fold finish, close the wandb logger.
    wandb.finish()

    # return the best acc score.
    return model_check_point.best_model_score.item()

@hydra.main(
        version_base=None,
        config_path='/workspace/Two_Stream_PyTorch/configs/',
        config_name='config.yaml',
        )
def init_params(config):

    #############
    # K Fold CV
    #############

    DATE = str(time.localtime().tm_mon) + str(time.localtime().tm_mday)
    DATA_PATH = config.data.data_path

    # set the version
    uniform_temporal_subsample_num = config.train.uniform_temporal_subsample_num
    clip_duration = config.train.clip_duration
    config.train.version = "_".join(
        [DATE, str(clip_duration), str(uniform_temporal_subsample_num)]
    )

    # output log to file
    log_path = (
        "/workspace/Two_Stream_PyTorch/logs/"
        + "_".join([config.train.version, config.model.model])
        + ".log"
    )
    sys.stdout = open(log_path, "w")

    # get the fold number
    fold_num = os.listdir(DATA_PATH)
    fold_num.sort()
    if "raw" in fold_num:
        fold_num.remove("raw")

    store_Acc_Dict = {}
    sum_list = []

    for fold in fold_num:
        #################
        # start k Fold CV
        #################

        logging.info("#" * 50)
        logging.info("Start %s" % fold)
        logging.info("#" * 50)

        config.train.train_path = os.path.join(DATA_PATH, fold)
        config.train.fold = fold

        Acc_score = train(config)

        store_Acc_Dict[fold] = Acc_score
        sum_list.append(Acc_score)

    logging.info("#" * 50)
    logging.info("different fold Acc:")
    logging.info(store_Acc_Dict)
    logging.info("Final avg Acc is: %s" % (sum(sum_list) / len(sum_list)))

if __name__ == '__main__':

    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
