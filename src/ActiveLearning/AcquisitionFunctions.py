import numpy as np
import torch
from scipy.special import xlogy
from scipy.stats import entropy


class ActiveLearningAcquisitions:
    def __init__(self, base=None):
        self.base = base

    def ShanonEntropy(self, predictions):
        """
        Input:
            predictions: Prediction tensor. Shape [batch_size, number_predictions, width, height, n_classes]
        Output:
            Shanon Entropy applied as the sum over the height and width of the image
        """
        mean_predictions = torch.mean(predictions, dim=1)  # Find mean predictions
        mean_predictions = mean_predictions.detach().cpu().numpy()
        # np.apply_along_axis(entropy,axis=-1,arr=mean_predictions).sum(axis=(1,2))
        return torch.tensor(-xlogy(mean_predictions, mean_predictions).sum(axis=(1, 2, 3)))

    def BALD(self, predictions):
        """
        Input:
            predictions: Prediction tensor. Shape [batch_size, number_predictions, width, height, n_classes]
        Output:
            BALD score Entropy the sum over the height and width of the image
        """
        mean_predictions = predictions.mean(dim=1)
        mean_predictions = mean_predictions.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()

        entropy_expected_preds = -xlogy(mean_predictions, mean_predictions).sum(axis=(1, 2, 3))
        expected_entropy = -np.mean(xlogy(predictions, predictions).sum(axis=(2, 3, 4)), axis=1)
        return torch.tensor(entropy_expected_preds - expected_entropy)

    def Random(self, unlabeled_pool, n=2):
        """
        Input:
            unlabeled_pool: array of idx of unlabeled images
            n: the number of images that should be random sampled
        Return:
            A list of the idx on the random sampled images from the
            unlabeled images pool
        """
        if len(unlabeled_pool) < n:
            return [unlabeled_pool[0]]
        return list(np.random.choice(unlabeled_pool, size=n, replace=False))

    def JensenDivergence(self, predictions):
        """
        Input:
            predictions: Prediction tensor. Shape [batch_size, number_predictions, width, height, n_classes]
        Output:
            JSD Divergence the sum over the height and width of the image
        """
        predictions = predictions.numpy()
        consenus_prob = np.mean(predictions, axis=1)
        consenus_prob = np.repeat(consenus_prob, repeats=predictions.shape[1], axis=0).reshape(
            predictions.shape
        )
        M = 0.5 * (predictions + consenus_prob)
        KL_PM = entropy(predictions, M, axis=-1).sum(axis=(2, 3))
        KL_QM = entropy(consenus_prob, M, axis=-1).sum(axis=(2, 3))
        JSD_values = (0.5 * (KL_PM + KL_QM)).mean(-1)
        return torch.tensor(JSD_values)

    def KLDivergence(self):
        pass

    def ApplyAcquisition(self, predictions, method, n=2):
        if method == "ShanonEntropy":
            return self.ShanonEntropy(predictions)
        elif method == "BALD":
            return self.BALD(predictions)
        elif method == "Random":
            return self.Random(predictions, n=n)
        elif method == "JensenDivergence":
            return self.JensenDivergence(predictions)
        # elif method == "KLDivergence":
        #    return self.KLDivergence(predictions)
