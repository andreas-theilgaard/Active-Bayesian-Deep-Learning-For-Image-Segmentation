# import torch
# from torchmetrics import JaccardIndex,Dice
# import torch.nn.functional as F


# masks =torch.load('masks.pth')
# pred = torch.load('test_pred.pth')

# dice = Dice(average='micro',ignore_index=0)
# masks.shape

# dice(pred,masks.squeeze(1))


# masks[0].squeeze(0).shape

# pred[0].argmax(0).shape


# torch.sum(pred[0].argmax(0)*masks[0].squeeze(0))/(torch.sum(pred[0].argmax(0))+torch.sum(masks[0].squeeze(0)))

# torch.equal(pred[0].argmax(0),masks[0].squeeze(0))

# torch.sum(pred[0].argmax(0)==masks[0].squeeze(0))/(torch.sum(pred[0].argmax(0))+torch.sum(masks[0].squeeze(0)))

# 3933/(64*64)

# (torch.sum(pred[0].argmax(0))+torch.sum(masks[0].squeeze(0)))


# pred[0].argmax(0)

# pred[4].argmax(0).unique()

# #################################
# def dice_coef_cpu(y_true, y_pred, smooth=1e-12):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = torch.sum(y_true_f*y_pred_f)
#     union = torch.sum(y_true_f) + torch.sum(y_pred_f)
#     return  ((2. * intersection + smooth)/(union + smooth)).item()

# masks = F.one_hot(masks.squeeze(1), 14)
# masks = torch.reshape(masks,shape=[masks.shape[0],masks.shape[1]*masks.shape[2],masks.shape[3]])

# # prediction
# pred = F.softmax(pred,dim=1)
# pred = torch.reshape(pred,shape=[pred.shape[0],pred.shape[2]*pred.shape[3],pred.shape[1]])

# torch.tensor([dice_coef_cpu(masks[j],pred[j]) for j in range(pred.shape[0])]).mean().item()

# masks = F.one_hot(masks.squeeze(1), 14)
# pred = F.softmax(pred,dim=1)
# pred = pred.permute(0,2,3,1)
# print(pred.shape,masks.shape)

# intersection = torch.sum(pred*masks,dim=(1,2,3))
# union = torch.sum(masks,dim=(1,2,3))+torch.sum(pred,dim=(1,2,3))
# smooth=1e-12
# torch.mean((2*intersection+smooth)/(union+smooth),dim=0)
