import torch
import numpy as np
from numpy import ndarray as array
from torch.nn.functional import softmax
from torch.nn.functional import one_hot


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

    def torch_their_dice(self, pred, mask):
        assert torch.is_tensor(pred) == True
        assert torch.is_tensor(mask) == True
        mask = mask.unsqueeze(1)
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask, dim=(1, 2, 3))
        union = torch.sum(mask, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3))
        smooth = 1e-12
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return torch.mean(dice)

    def pixel_acc(self, yhat, y_true):
        assert torch.is_tensor(yhat) == True
        assert torch.is_tensor(y_true) == True
        if yhat.min().item() < 0.0:
            yhat = (torch.sigmoid(yhat) >= 0.50).float()

        TP = (yhat.flatten() * y_true.flatten()).sum()
        FN = y_true[yhat == 0].sum()
        FP = yhat[y_true == 0].sum()
        TN = yhat.numel() - TP - FN - FP

        pixel_acc = (TP + TN) / (TP + FN + FP + TN)
        return pixel_acc


class Calibration_Scoring_Metrics:
    def __init__(self, nbins, multiclass, device, is_prob=False):

        self.nbins = nbins
        self.multiclass = multiclass
        self.device = device
        self.is_prob = is_prob

    def brier_score(self, y_hat, y_true, sample_wise=True):
        if isinstance(y_hat, array) and isinstance(y_true, array):
            y_hat, y_true = torch.tensor(y_hat).to(self.device), torch.tensor(y_true).to(
                self.device
            )
        if self.multiclass:
            n_classes = y_hat.shape[1]
            p = softmax(y_hat, dim=1)
            y_true = one_hot(y_true.to(dtype=torch.int64), n_classes).permute(0, 3, 1, 2)
            brier = torch.mean(
                torch.sum(((p - y_true) ** 2), dim=1).view(y_hat.shape[0], -1), dim=1, keepdim=True
            )
            if not sample_wise:
                brier = torch.mean(brier)
            return brier.detach().cpu().numpy().ravel()
        else:
            p = torch.sigmoid(y_hat)
            # brier = torch.mean(((p - y_true) ** 2).view(y_hat.shape[0], -1), dim=1, keepdim=True)
            brier = torch.mean(((p - y_true) ** 2).reshape(y_hat.shape[0], -1), dim=1, keepdim=True)
            if not sample_wise:
                brier = torch.mean(brier)
            return brier.detach().cpu().numpy().ravel()

    def get_max_prob(self, y_hat):
        # if self.multiclass:
        #    print("not implemented yet")
        # return torch.max(torch.softmax(y_hat,dim=1),dim=1)[0]
        # else:
        return torch.sigmoid(y_hat).squeeze(1)

    def get_prediction(self, y_hat):
        return y_hat >= 0.5

    def Calibration_Errors(self, y_hat, y_true, num_bins, is_prob=False):
        confidence = y_hat
        if is_prob == False:
            confidence = self.get_max_prob(y_hat)
        bin_bounds = np.linspace(0, 1, num_bins + 1)
        bin_ace = []  # Absolut calibration error
        bin_accuracy = []  # Accuracy in bin
        bin_confidence = []  # Confidence in bin

        ECE = 0  # ECE

        for i in range(num_bins):
            mask = (confidence > bin_bounds[i]) & (confidence < bin_bounds[i + 1])
            if torch.any(mask):
                bin_acc = torch.mean(
                    (
                        (self.get_prediction(y_hat)[mask]).float()
                        == (y_true.unsqueeze(0)[mask]).float()
                    ).float()
                )  # calculates accuracy in i'th bin
                bin_conf = torch.mean(confidence[mask].float())  # calculates confidence in i'th bin

                # absoulut calibration
                ace = abs(bin_conf - bin_acc)  # *****
                ECE += torch.mean(mask.float()) * ace

                # store values
                bin_ace.append(ace.item())
                bin_accuracy.append(bin_acc.item())
                bin_confidence.append(bin_conf.item())
            else:
                bin_ace.append(0)
                bin_accuracy.append(0)
                bin_confidence.append(0)
        MCE = np.max(np.array(bin_ace))
        out = {"ECE": ECE.item(), "MCE": MCE}
        return out
        # return ECE.item(), MCE,np.array(bin_ace), np.array(bin_accuracy), np.array(bin_confidence)

    def NLL(self, y_hat, y_true, sample_wise=True):
        if not self.multiclass:
            if isinstance(y_hat, array) and isinstance(y_true, array):
                y_hat, y_true = torch.tensor(y_hat).to(self.device), torch.tensor(y_true).to(
                    self.device
                )
            # y_hat,y_true = y_hat.ravel(),y_true.ravel()
            p = torch.sigmoid(y_hat)
            # nll = -torch.mean(torch.sum(torch.log(p)*y_true.unsqueeze(1),dim=1).view(y_hat.shape[0],-1),dim=1,keepdim=True)
            # nll = -torch.mean((y_true*torch.log(p) + (1-y_true)*torch.log(1-p)))
            nll = -torch.mean(
                (
                    y_true.unsqueeze(1) * torch.log(p)
                    + (1 - y_true.unsqueeze(1)) * torch.log(1 - p)
                ).view(y_hat.shape[0], -1),
                dim=1,
                keepdim=True,
            )
            if not sample_wise:
                nll = torch.mean(nll)
            return nll.detach().cpu().numpy().ravel()
        else:
            if torch.is_tensor(y_hat) and torch.is_tensor(y_true):
                p = softmax(y_hat, dim=1)
                y_true = one_hot(y_true, y_hat.shape[1]).permute(0, 3, 1, 2)
                nll = -torch.mean(
                    torch.sum(torch.log(p) * y_true, dim=1).view(y_hat.shape[0], -1),
                    dim=1,
                    keepdim=True,
                )
                if not sample_wise:
                    nll = torch.mean(nll)
                return nll.detach().cpu().numpy().ravel()

    def metrics(self, y_hat, y_true, sample_wise=True):
        """
        Samplewise get metrics for each sample in batch
        """
        NLL = self.NLL(y_hat, y_true, sample_wise=sample_wise)
        brier_score = self.brier_score(y_hat, y_true, sample_wise=sample_wise)
        return {"NLL": list(NLL), "brier_score": list(brier_score)}
        # calib_metrics = [self.Calibration_Errors(y_hat[x],y_true[x],num_bins=self.nbins,is_prob=self.is_prob) for x in range(y_hat.shape[0])]
        # return {'NLL':list(NLL),
        #        'brier_score':list(brier_score),
        #        'ECE':[calib_metrics[x]['ECE'] for x in range(len(calib_metrics))],
        #        'MCE':[calib_metrics[x]['MCE'] for x in range(len(calib_metrics))]
        #        }


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
