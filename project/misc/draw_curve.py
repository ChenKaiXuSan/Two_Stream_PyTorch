# %%
import torch 
import torch.nn.functional as F
import os 
import sys

sys.path.append('/workspace/Two_Stream_PyTorch/project')

from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from dataloader.data_loader import WalkDataModule
from pytorchvideo.transforms.functional import uniform_temporal_subsample

from IPython.display import clear_output

clear_output()

from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

import torchmetrics

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pytorch_lightning import Trainer

# %%
# define the metrics.
_accuracy = torchmetrics.classification.BinaryAccuracy()
_precision = torchmetrics.classification.BinaryPrecision()
_binary_recall = torchmetrics.classification.BinaryRecall()
_binary_f1 = torchmetrics.classification.BinaryF1Score()

_confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix()

_aucroc = torchmetrics.classification.BinaryAUROC()

# %%
class opt:
    num_workers = 8
    batch_size = 32

    model = "single"
    model_type = "single"
    fusion = "sum"

    img_size = 224
    lr=0.0001
    model_class_num = 1
    model_depth = 50

    transfor_learning = True
    pre_process_flag = True

    data_path = "/workspace/data/dataset/"

# DATA_PATH = "/workspace/data/segmentation_dataset_512"
DATA_PATH = "/workspace/data/split_pad_dataset_512"

# %%
def get_best_ckpt(length: str, frame: str, fold: str):

    ckpt_path = '/workspace/Two_Stream_PyTorch/logs/single/'
    version = '515_1_31_single_sum_loss'
    ckpt_path_list = os.listdir(ckpt_path)
    ckpt_path_list.sort()

    final_ckpt_list = [] 

    for i in ckpt_path_list:
        if version in i:
            final_ckpt_list.append(i)

    final_ckpt_list.sort()
    
    for name in final_ckpt_list:
        if length in name.split('_') and frame in name.split('_'):
            ckpt = name

    ckpt = os.path.join(ckpt_path, ckpt, fold, 'checkpoints')
    
    Acc = 0.0

    ckpt_list = os.listdir(ckpt)
    ckpt_list.sort()
    ckpt_list.remove('last.ckpt')

    for num, l in enumerate(ckpt_list):
        flow_acc = l[-11:-5] # flow acc
        rgb_acc = l.split('-')[2].split('=')[1] # rgb acc

        if float(rgb_acc) > float(Acc):
            Acc = rgb_acc
            NUM = num

    return os.path.join(ckpt, ckpt_list[NUM])

# ckpt_path = get_best_ckpt('1', '31', 'fold0')

# %%
from models.optical_flow import Optical_flow

optical_flow_model = Optical_flow()

def single_logic(label: torch.Tensor, video: torch.Tensor, model):

    # pred the optical flow base RAFT
    video_flow = optical_flow_model.process_batch(video)  # b, c, t, h, w

    b, c, t, h, w = video.shape

    single_img = video[:, :, :-1, :].reshape(-1, 3, h, w)
    single_flow = video_flow.contiguous().view(-1, 2, h, w)

    with torch.no_grad():
        pred_video_rgb = model.model_rgb(single_img)
        pred_video_flow = model.model_flow(single_flow)

    return pred_video_rgb, pred_video_flow

def multi_logic(label: torch.Tensor, video: torch.Tensor, model):

    # pred the optical flow base RAFT
    video_flow = optical_flow_model.process_batch(video) # b, c, t, h, w

    with torch.no_grad():
        pred_video_rgb = model.model_rgb(video)
        pred_video_flow = model.model_flow(video_flow)

    return pred_video_rgb, pred_video_flow

def multi_single_logic(label: torch.Tensor, video: torch.Tensor, model):
        
    single_label = label.repeat_interleave(video.size()[2] - 1)

    # pred the optical flow base RAFT
    video_flow = optical_flow_model.process_batch(video) # b, c, t, h, w
    # extract 16 for 3D CNN
    video = uniform_temporal_subsample(video[:, :, :-1, :].cpu(), 16).cuda()

    b, c, t, h, w = video.shape

    single_flow = video_flow.contiguous().view(-1, 2, h, w)

    with torch.no_grad():
        pred_video_rgb = model.model_rgb(video)
        pred_video_flow = model.model_flow(single_flow)

    return pred_video_rgb, pred_video_flow
# %%
def get_inference(test_data, model):
        
    pred_rgb_list = []
    pred_flow_list = []
    label_list = []

    for i, batch in enumerate(test_data):

        # input and label
        video = batch['video'].detach().cuda() # b, c, t, h, w
        label = batch['label'].detach().cuda() # b, class_num

        model.cuda().eval()

        if opt.model == 'multi':
            pred_rgb, pred_flow = multi_logic(label, video, model)
        elif opt.model == 'single':
            # not use the last frame
            label = label.repeat_interleave(video.size()[2] - 1).cuda()
            pred_rgb, pred_flow = single_logic(label, video, model)
        elif opt.model == 'multi_single':
            pred_rgb, pred_flow = multi_single_logic(label, video, model)

        # when torch.size([1]), not squeeze.
        if pred_rgb.size()[0] != 1 or len(pred_rgb.size()) != 1 :
            preds_rgb = pred_rgb.squeeze(dim=-1)
            preds_flow = pred_flow.squeeze(dim=-1)

            preds_rgb_sigmoid = torch.sigmoid(preds_rgb)
            preds_flow_sigmoid = torch.sigmoid(preds_flow)
        else:
            preds_rgb_sigmoid = torch.sigmoid(pred_rgb)
            preds_flow_sigmoid = torch.sigmoid(pred_flow)

        pred_rgb_list.append(preds_rgb_sigmoid.tolist())
        pred_flow_list.append(preds_flow_sigmoid.tolist())
        label_list.append(label.tolist())

    total_pred_rgb_list = []
    total_pred_flow_list = []
    total_label_list = []

    for i in pred_rgb_list:
        for number in i:
            total_pred_rgb_list.append(number)

    for i in pred_flow_list:
        for number in i:
            total_pred_flow_list.append(number)

    for i in label_list:
        for number in i: 
            total_label_list.append(number)

    return total_pred_rgb_list, total_pred_flow_list, total_label_list

# %%
if opt.model == 'multi':
    VIDEO_LENGTH = ['1']
    VIDEO_FRAME = ['17']
else:
    VIDEO_LENGTH = ['1']
    VIDEO_FRAME = ['31']

# %%
fold_num = os.listdir(DATA_PATH)
fold_num.sort()
fold_num.remove('raw')

symbol = '_'

one_condition_pred_rgb_list = []
one_condition_pred_flow_list = []
one_condition_label_list = []

total_pred_rgb_list = []
total_pred_flow_list = []
total_label_list = []

for length in VIDEO_LENGTH:

    for frame in VIDEO_FRAME:

        for fold in fold_num:

            opt.train_path = os.path.join(DATA_PATH, fold)

            #################
            # start k Fold CV
            #################
            
            opt.clip_duration = int(length)
            opt.uniform_temporal_subsample_num = int(frame)
           
            ckpt_path = get_best_ckpt(length, frame, fold)

            print('#' * 50)
            print('Strat %s, %s length, %s frames' % (fold, length, frame))
            print('the data path: %s' % opt.train_path)
            print('ckpt: %s' % ckpt_path)
            model = WalkVideoClassificationLightningModule(opt).load_from_checkpoint(ckpt_path, hparams=opt)

            data_module = WalkDataModule(opt)
            data_module.setup()

            test_data = data_module.test_dataloader()

            pred_rgb_list, pred_flow_list, label_list = get_inference(test_data, model)

            one_condition_pred_rgb_list.append(pred_rgb_list)
            one_condition_pred_flow_list.append(pred_flow_list)
            one_condition_label_list.append(label_list)

        # total 5 fold pred and label
        for i in one_condition_pred_rgb_list:
            for number in i:
                total_pred_rgb_list.append(number)

        for i in one_condition_pred_flow_list:
            for number in i: 
                total_pred_flow_list.append(number)

        for i in one_condition_label_list:
            for number in i: 
                total_label_list.append(number)

        pred_rgb = torch.tensor(total_pred_rgb_list)    
        pred_flow = torch.tensor(total_pred_flow_list)
        label = torch.tensor(total_label_list)

        print('*' * 100)

        # video rgb metrics
        print('the result of %ss %sframe:' % (length, frame))
        print('accuracy: %s' % _accuracy(pred_rgb, label))
        print('precision: %s' % _precision(pred_rgb, label))
        print('_binary_recall: %s' % _binary_recall(pred_rgb, label))
        print('_binary_f1: %s' % _binary_f1(pred_rgb, label))
        print('_aurroc: %s' % _aucroc(pred_rgb, label))
        print('_confusion_matrix: %s' % _confusion_matrix(pred_rgb, label))
        print('#' * 50)

        if opt.model == 'multi_single':
            label = label.repeat_interleave(30)

        # video optical metrics
        print('the result of %ss %sframe:' % (length, frame))
        print('accuracy: %s' % _accuracy(pred_flow, label))
        print('precision: %s' % _precision(pred_flow, label))
        print('_binary_recall: %s' % _binary_recall(pred_flow, label))
        print('_binary_f1: %s' % _binary_f1(pred_flow, label))
        print('_aurroc: %s' % _aucroc(pred_flow, label))
        print('_confusion_matrix: %s' % _confusion_matrix(pred_flow, label))
        print('#' * 50)

# %%
# sum the two pred to predict the final metrics
total_preds = (pred_rgb + pred_flow) /2

print('*' * 100)

# video rgb metrics
print('the result of %ss %sframe:' % (length, frame))
print('accuracy: %s' % _accuracy(total_preds, label))
print('precision: %s' % _precision(total_preds, label))
print('_binary_recall: %s' % _binary_recall(total_preds, label))
print('_binary_f1: %s' % _binary_f1(total_preds, label))
print('_aurroc: %s' % _aucroc(total_preds, label))
print('_confusion_matrix: %s' % _confusion_matrix(total_preds, label))
print('#' * 50)
# %%
# reduce_pred_flow = []
# a, b = 0, 30
# for i in range(pred_flow.size()[0]//30):
#     reduce_pred_flow.append(sum(pred_flow[a:b])/30)
#     a += 30
#     b += 30

# reduce_pred_flow = torch.sigmoid(torch.tensor(reduce_pred_flow))

# # %%
# reduce_label = []
# a, b = 0, 30
# for i in range(label.size()[0]//30):
#     reduce_label.append(sum(label[a:b])/30)
#     a +=30
#     b += 30
# reduce_label = torch.tensor(reduce_label)

# # %%
# # sum the two pred to predict the final metrics
# total_preds = (reduce_pred_flow + pred_rgb) /2

# print('*' * 100)

# # video rgb metrics
# print('the result of %ss %sframe:' % (length, frame))
# print('accuracy: %s' % _accuracy(total_preds, reduce_label))
# print('precision: %s' % _precision(total_preds, reduce_label))
# print('_binary_recall: %s' % _binary_recall(total_preds, reduce_label))
# print('_binary_f1: %s' % _binary_f1(total_preds, reduce_label))
# print('_aurroc: %s' % _aucroc(total_preds, reduce_label))
# print('_confusion_matrix: %s' % _confusion_matrix(total_preds, reduce_label))
# print('#' * 50)
# %%
# sum the two pred to predict the final metrics
x = pred_rgb.repeat_interleave(30)
total_preds = (x + pred_flow) /2

print('*' * 100)

# video rgb metrics
print('the result of %ss %sframe:' % (length, frame))
print('accuracy: %s' % _accuracy(total_preds, label))
print('precision: %s' % _precision(total_preds, label))
print('_binary_recall: %s' % _binary_recall(total_preds, label))
print('_binary_f1: %s' % _binary_f1(total_preds, label))
print('_aurroc: %s' % _aucroc(total_preds, label))
print('_confusion_matrix: %s' % _confusion_matrix(total_preds, label))
print('#' * 50)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

cm = _confusion_matrix(total_preds, label).to(float)

total = sum(sum(cm))

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cm[i][j] = float(cm[i][j] / total)

ax = sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=['ASD', 'non_ASD'], yticklabels=['ASD', 'non_ASD'])

ax.set_title('Confusion Matrix')
ax.set(xlabel="pred class", ylabel="ground truth")
ax.xaxis.tick_top()
plt.show()

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay


fpr, tpr, _ = roc_curve(label, total_preds, pos_label=1)
roc_auc = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="proposed method").plot()
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC curves")
plt.legend()
plt.show()


# %%
data_module = WalkDataModule(opt)
data_module.setup()

test_data = data_module.test_dataloader()

# %%
video = next(iter(test_data))
# %%
video_rgb = video['video'].cuda()
optical_flow_model = Optical_flow()

# pred the optical flow base RAFT
video_flow = optical_flow_model.process_batch(video_rgb)
# %%
one_video = video_rgb[2]
# %%
from matplotlib import pyplot as plt 

c, f, h, w = one_video.shape
for i in range(f):

    plt.imshow(one_video[:, i, :].cpu().permute(1,2,0))
    plt.show()
# %%
from torchvision.utils import flow_to_image

flow_imgs = flow_to_image(video_flow[2].permute(1,0,2,3))

f, c, h, w = flow_imgs.shape

for i in range(f):
    plt.imshow(flow_imgs[i].cpu().permute(1,2,0))
    plt.show()
# %%
