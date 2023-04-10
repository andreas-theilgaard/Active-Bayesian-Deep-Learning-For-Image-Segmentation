import torch


def binarize(predictions):
    if len(predictions.shape) == 4:
        # predictions input shape: [batch_size, n_classes=1, img_width,img_height]
        return torch.cat([1 - predictions, predictions], dim=1)
    elif len(predictions.shape) == 5:
        # predictions input shape: [batch_size, n_predictions, img_width,img_height,n_classes=1]
        return torch.cat([1 - predictions, predictions], dim=4)


def unbinarize(predictions):
    if len(predictions.shape) == 5:
        # In: [batch_size, n_predictions, img_width,img_height,n_classes=2]
        # Out: [batch_size, n_predictions, img_width,img_height,n_classes=1]
        return predictions[:, :, :, :, 1].unsqueeze(4)
    elif len(predictions.shape) == 4:
        # In: [batch_size, n_classes=2, img_width,img_height]
        # Out: [batch_size, n_classes=1, img_width,img_height]
        return predictions[:, 1, :, :].unsqueeze(1)
