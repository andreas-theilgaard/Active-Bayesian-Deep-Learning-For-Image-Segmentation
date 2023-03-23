import numpy as np
import torch
from scipy.special import xlogy, kl_div
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

# def binarize(predictions):
#     if len(predictions.shape) == 4:
#         return torch.cat([predictions,1-predictions],dim=3)
#     elif len(predictions.shape)==5:
#         return torch.cat([predictions,1-predictions],dim=4)

# predictions = torch.stack([binarize(torch.sigmoid(torch.rand(4,64,64,1))) for _ in range(10)],dim=1)


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
        # np.apply_along_axis(entropy,axis=-1,arr=mean_predictions).sum(axis=(1,2))
        return -xlogy(mean_predictions, mean_predictions).sum(dim=(1, 2, 3))

    def BALD(self, predictions):
        """
        Input:
            predictions: Prediction tensor. Shape [batch_size, number_predictions, width, height, n_classes]
        Output:
            BALD score Entropy the sum over the height and width of the image
        """
        mean_predictions = predictions.mean(dim=1)
        entropy_expected_preds = -xlogy(mean_predictions, mean_predictions).sum(dim=(1, 2, 3))
        expected_entropy = -torch.mean(xlogy(predictions, predictions).sum(dim=(2, 3, 4)), dim=1)
        return entropy_expected_preds - expected_entropy

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

    def JensenDivergence(self):
        pass

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
        elif method == "KLDivergence":
            return self.KLDivergence(predictions)


# Shanon Entropy
# Find Mean

# def ShanonEntropy(predictions):
#     """
#     Input:
#         predictions: Prediction tensor. Shape [batch_size, number_predictions, width, height, n_classes]
#     Output:
#         Shanon Entropy applied as the sum over the height and width of the image
#     """
#     mean_predictions = torch.mean(predictions,dim=1) # Find mean predictions
#     #np.apply_along_axis(entropy,axis=-1,arr=mean_predictions).sum(axis=(1,2))
#     return -xlogy(mean_predictions,mean_predictions).sum(dim=(1,2,3))
# # BALD
# def BALD(predictions):
#     """
#     Input:
#         predictions: Prediction tensor. Shape [batch_size, number_predictions, width, height, n_classes]
#     Output:
#         BALD score Entropy the sum over the height and width of the image
#     """
#     mean_predictions = predictions.mean(dim=1)
#     entropy_expected_preds = - xlogy(mean_predictions,mean_predictions).sum(dim=(1,2,3))
#     expected_entropy = -torch.mean(xlogy(predictions,predictions).sum(dim=(2,3,4)),dim=1)
#     return entropy_expected_preds-expected_entropy

# def KL_Divergence(predictions):
#     """
#     Input:
#         predictions: Prediction tensor. Shape [batch_size, number_predictions, width, height, n_classes]
#     Output:
#         BALD score Entropy the sum over the height and width of the image
#     """
#     predictions = predictions.numpy()
#     consenus_prob = np.mean(predictions,axis=1)
#     consensus_prob_formatted = np.repeat(consenus_prob,repeats=predictions.shape[1],axis=0).reshape(predictions.shape)
#     return torch.tensor((entropy(predictions,consensus_prob_formatted,axis=-1).sum(axis=(2,3))).mean(axis=1))

# def Jensen_Divergence(predictions):
#     predictions = predictions.numpy()
#     consenus_prob = np.mean(predictions,axis=1)
#     consensus_prob_formatted = np.repeat(consenus_prob,repeats=predictions.shape[1],axis=0).reshape(predictions.shape)
#     return torch.tensor((jensenshannon(predictions,consensus_prob_formatted,axis=-1).sum(axis=(2,3))).mean(axis=1))


# ShanonEntropy(predictions)
# BALD(predictions)
# KL_Divergence(predictions)
# Jensen_Divergence(predictions)

# predictions_ = predictions.numpy()
# consenus_prob = np.mean(predictions_,axis=1)
# consensus_prob_formatted = np.repeat(consenus_prob,repeats=predictions_.shape[1],axis=0).reshape(predictions_.shape)


# np.mean(jensenshannon(predictions_,consensus_prob_formatted,axis=-1).sum(axis=(2,3)),axis=-1)

# predictions = predictions.numpy()
# consenus_prob = np.mean(predictions,axis=1)
# consensus_prob_formatted = np.repeat(consenus_prob,repeats=predictions.shape[1],axis=0).reshape(predictions.shape)
# consensus_prob_formatted.shape
# jensenshannon(predictions,consensus_prob_formatted,axis=-1)


# def KLDivergence(p,q):
#     return (xlogy(p,p/(q)).sum((2,3,4)))

# (xlogy(predictions,predictions/(consensus_prob_formatted)).sum((2,3,4)))

# M= (predictions+consensus_prob_formatted)/2

# Left = (1/2)*KLDivergence(predictions,M)
# Right = (1/2)*KLDivergence(consensus_prob_formatted,M)
# (Left+Right).shape

# a = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])
# b = np.array([[13, 14, 15, 16],[17, 18, 19, 20],[21, 22, 23, 24]])

# M= (a+b)/2

# Left = (xlogy(a,a/M).sum(1))*(1/2)
# Right = (xlogy(b,b/M).sum(1))*(1/2)
# Left+Right

# jensenshannon(a,b,axis=1)

# np.mean(jensenshannon(predictions,consensus_prob_formatted,axis=(2,3,4)),axis=1)


# def BALDOther(predictions):
#     """
#     predictions: shape = [batch_size, Number_Predictions, 64,64, 2]
#     """
#     #
#     H1 = np.apply_along_axis(entropy,axis=3,arr=predictions.mean(axis=1)).sum(axis=(1,2))#Consensus_Probability
#     H2 = np.apply_along_axis(entropy,arr=predictions,axis=4).mean(axis=1).sum(axis=(1,2))
#     return H1-H2

# def KL_Divergence(predictions):
#     """
#     Input:
#         predictions: Prediction tensor. Shape [batch_size, number_predictions, width, height, n_classes]
#     Output:
#         BALD score Entropy the sum over the height and width of the image
#     """
#     predictions = predictions.numpy()
#     consenus_prob = np.mean(predictions,axis=1)
#     consensus_prob_formatted = np.repeat(consenus_prob,repeats=predictions.shape[1],axis=0).reshape(predictions.shape)
#     np.mean(entropy(predictions,consensus_prob_formatted,axis=-1),axis=1)
#     return torch.tensor((entropy(predictions,consensus_prob_formatted,axis=-1).sum(axis=(2,3))).mean(axis=1))


# consensus_prob_formatted = np.repeat(consenus_prob,repeats=predictions.shape[1],axis=0).reshape(predictions.shape)
# return np.mean(entropy(predictions,consensus_prob_formatted,axis=-1),axis=1)


# def Jensen_Divergence(predictions):
#     predictions = predictions.numpy()
#     consenus_prob = np.mean(predictions,axis=1)
#     consensus_prob_formatted = np.repeat(consenus_prob,repeats=predictions.shape[1],axis=0).reshape(predictions.shape)
#     return np.mean(jensenshannon(predictions,consensus_prob_formatted,axis=-1),axis=1)
# Kl_Divergence(predictions)
# Jensen_Divergence(predictions)

# BALDOther(predictions)
# BALD(predictions)


# import numpy as np
# A = np.arange(0,100)
# np.random.uniform(A,size=2)
# np.random.sample(A,size=2)
# list(np.random.choice(A,size=2,replace=False))


# def Random(unlabeled_pool,n=2):
#     return


# def test_acquisition_functions():
#     pass


# if __name__ == "__main__":
