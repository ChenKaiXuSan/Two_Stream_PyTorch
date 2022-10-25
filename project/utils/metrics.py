# %%
import torch
import torchmetrics


def get_Accuracy(num_class):

    accuracy = torchmetrics.Accuracy(
        num_classes=num_class,
        average='micro',
        multiclass=False
        # top_k=1
    )

    return accuracy


def get_Precision(num_class):

    precision = torchmetrics.Precision(
        num_classes=num_class,
        average='micro',
        multiclass=False,
    )

    return precision


def get_Dice():
    dice = torchmetrics.Dice(
        num_classes=2,
        average='micro',

    )
    return dice


# def get_Average_precision(num_class):

#     average_precision = torchmetrics.AveragePrecision(
#         # num_classes=num_class,
#         pos_label=0,
#         average=None
#     )

#     return average_precision


def get_Precision_Recall():
    precision_recall = torchmetrics.PrecisionRecallCurve(
        num_classes=2,
        pos_label=1
    )

    return precision_recall


def get_AUC():
    AUC = torchmetrics.AUC(
        reorder=True
    )
    return AUC


def get_F1Score():
    F1_Score = torchmetrics.F1Score(
        num_classes=2,

    )
    return F1_Score


def get_Confusion_Matrix():
    confusion_matrix = torchmetrics.ConfusionMatrix(
        num_classes=2,
        multilabel=False
    )

    return confusion_matrix

# %%

# preds = torch.tensor([1, 0.4, 0.8, 0.4])
# target = torch.tensor([1, 1, 0, 0])

# dice = get_Dice()
# accuracy = get_Accuracy(1)
# average_precision = get_Average_precision(1)
# precision = get_Precision(1)
# confusion_matrix = get_Confusion_Matrix()

# # result = accuracy(preds, target)
# acc = accuracy(preds, target)
# confusion  = confusion_matrix(preds, target)
# result = precision(preds, target)

# print(acc, result, confusion)
# %%
