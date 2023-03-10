import torch


class SegmentationMetrics:
    def __init__(self, multiclass=False):
        self.multiclass = multiclass

    def Dice_Coef(self, yhat, y_true):
        assert torch.is_tensor(yhat) == True
        assert torch.is_tensor(y_true) == True

        if yhat.min().item() < 0.0:
            yhat = (torch.sigmoid(yhat) >= 0.50).float()

        intersection = (yhat.flatten() * y_true.flatten()).sum()
        union = yhat.flatten().sum() + y_true.flatten().sum()
        numeric_stability = 1e-8
        return (2 * intersection) / (union + numeric_stability)

    def Dice_Coef_Confusion(self, yhat, y_true):
        assert torch.is_tensor(yhat) == True
        assert torch.is_tensor(y_true) == True

        if yhat.min().item() < 0.0:
            yhat = (torch.sigmoid(yhat) >= 0.50).float()

        TP = (yhat.flatten() * y_true.flatten()).sum()
        FN = y_true[yhat == 0].sum()
        FP = yhat[y_true == 0].sum()
        TN = yhat.numel() - TP - FN - FP
        numeric_stability = 1e-8
        return (2 * TP) / (2 * TP + FN + FP + numeric_stability)

    def IOU_(self, yhat, y_true):
        assert torch.is_tensor(yhat) == True
        assert torch.is_tensor(y_true) == True

        if yhat.min().item() < 0.0:
            yhat = (torch.sigmoid(yhat) >= 0.50).float()

        TP = (yhat.flatten() * y_true.flatten()).sum()
        FN = y_true[yhat == 0].sum()
        FP = yhat[y_true == 0].sum()
        TN = yhat.numel() - TP - FN - FP
        # numeric_stability = 1e-8
        IOU = TP / (TP + FN + FP)
        return IOU


# yhat = torch.tensor([0,1,0,1])
# y_true = torch.tensor([0,0,1,1])

# metrics = SegmentationMetrics()

# metrics.Dice_Coef(yhat,y_true)
# metrics.Dice_Coef_Confusion(yhat,y_true)


# import numpy as np
# import torch
# from sklearn.metrics import f1_score
# from torchmetrics.classification import BinaryF1Score # Use BinaryF1 Score seems to work the best
# from torchmetrics import Dice
# y_true = np.array([[0,1,1,0],[0,1,1,0],[0,1,1,0],[1,1,0,1]])
# y_pred = np.array([[0,0,0,0],[0,1,1,0],[0,0,0,0],[1,0,0,1]])
# y_true1 = np.array([[1,1,1,0],[1,1,0,0],[0,0,1,1],[0,0,1,0]])
# y_pred1 = np.array([[1,0,1,0],[0,1,1,0],[1,0,1,1],[0,1,0,1]])

# np.vstack((y_true,y_true1)).shape

# yhat=np.array([y_pred,y_pred1])
# y=np.array([y_true,y_true1])

# f1_ = BinaryF1Score()
# Dice_score = Dice(average='micro',num_classes=1,multiclass=False)
# f1_score(y, yhat, average='micro')
# f1_(torch.tensor(yhat),torch.tensor(y))
# Dice_score(torch.tensor(yhat),torch.tensor(y))

# (0.58+0.61)/2

# yhat=torch.tensor(yhat)
# y=torch.tensor(y)

# intersection=(yhat.flatten()*y.flatten()).sum()
# union = yhat.flatten().sum()+y.flatten().sum()

# (2*intersection)/(union)


# f1_ = BinaryF1Score()
# Dice_score = Dice(average='micro',num_classes=1,multiclass=False)
# f1_score(y_true, y_pred, average='micro')
# f1_(torch.tensor(y_pred),torch.tensor(y_true))
# Dice_score(torch.tensor(y_pred),torch.tensor(y_true))

# 10/17
# f1_score(y_true1, y_pred1, average='micro')
# f1_(torch.tensor(y_pred1),torch.tensor(y_true1))
# Dice_score(torch.tensor(y_pred1),torch.tensor(y_true1))
