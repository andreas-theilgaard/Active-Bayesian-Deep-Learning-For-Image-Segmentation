import torch
from src.models.model import UNET, init_weights
from src.data.dataloader import train_split
import torch.nn.functional as F
import numpy as np
from math import *
from src.models.laplace_utils import compute_covariance, optimize_prior
from torch.distributions.multivariate_normal import MultivariateNormal

# from backpack import extend, backpack, extensions


class LaplacePredict:
    def __init__(self, method, n_samples=50):
        self.method = method
        self.n_samples = n_samples

    def probit_predict(self, X_val, Sigma, model, W, bias, validate_args):
        """
        model: The model object used for predictions
        X_val: torch tensor, representing the tensor that should be predicted
        Sigma: The Variance Covariance Matrix
        W: is the model weights of the last layer
        bias: is the model bias of the last layer if any
        validate_args: Bool whether correctness of the argument should be tested.
                        When testing consider setting it equal to 'True'
                        But in production set it equal to 'False' for faster computations
        """
        with torch.no_grad():
            phi = model(X_val, features=True)
        m = F.conv2d(phi, weight=W, bias=bias)
        if validate_args:
            assert m[-1, -1, -1, -1].item() == model(X_val)[-1, -1, -1, -1].item()
        # phi = phi.detach().cpu().numpy()

        # Use torch einsum instead
        Sigma = torch.tensor(Sigma)
        V1 = phi.reshape(phi.shape[0] * 64 * 64, 64)
        V2 = torch.matmul(V1, Sigma)
        V3 = torch.matmul(V2, V1.T)
        V = torch.diag(V3)

        m_reshaped = m.flatten()
        assert m[0][0][0][0] == m_reshaped[0]
        assert m[0][0][0][48] == m_reshaped[48]
        predictive = torch.sigmoid(m_reshaped / (torch.sqrt(1 + pi / 8 * V)))

        predictive = predictive.reshape(phi.shape[0], 1, 64, 64)
        return predictive

    def MC_predict(self, X_val, Sigma, model, W, bias, device):
        """
        Returns:
            predictions: [batch_size,n_samples,img_height,img_width,n_classes]
        """
        with torch.no_grad():
            phi = model(X_val, features=True)  # Features Before Last Layer
        Sigma = torch.tensor(Sigma).to(device=device)
        posterior_distribution = MultivariateNormal(W.flatten(), Sigma)
        # Sample From Posterior n times

        predictions = []
        for _ in range(self.n_samples):
            W_i = posterior_distribution.rsample().view(
                W.shape
            )  # i'th weight sample from posterior
            pred_i = F.conv2d(phi, weight=W_i, bias=bias)  # MAP
            predictions.append(pred_i.permute(0, 2, 3, 1))
        #
        predictions = torch.stack(predictions, dim=1)
        return predictions

    def Predict(self, X_val, Sigma, model, W, bias, device, validate_args=False):
        if self.method == "Probit":
            return self.probit_predict(X_val, Sigma, model, W, bias, validate_args)
        elif self.method == "MC":
            return self.MC_predict(X_val, Sigma, model, W, bias, device=device)


class Laplace:
    def __init__(
        self,
        model,
        binary,
        Train_Data,
        hessian_method,
        method,
        n_samples,
        prior=1,
        validate_args=True,
        device=None,
        dataset=None,
    ):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.binary = binary
        self.Train_Data = Train_Data
        self.hessian_method = hessian_method
        self.method = method
        self.n_samples = n_samples
        self.prior = prior.to(device=self.device)
        self.validate_args = validate_args

        self.opt_prior = None
        self.Sigma = None  # Variance-Covariance Matrix
        self.Train_Image = None
        self.Train_Target = None
        self.Train_Predictions = None
        self.Train_Predictions_idx = None

        self.LaplacePrediction = LaplacePredict(method=self.method, n_samples=self.n_samples)

        Weights_Last_Layer = list(model.parameters())[-2:]
        self.W = Weights_Last_Layer[0]
        self.bias = Weights_Last_Layer[1]

    def validate_data(self, data):
        if isinstance(data, dict):
            return (data["images"], data["masks"], data["idx"])
        else:
            imgs = []
            targets = []
            prediction_idx = []
            for batch in data:
                images, masks, idx = batch
                images = images.unsqueeze(1) if self.dataset != "warwick" else images
                images = images.to(device=self.device, dtype=torch.float32)
                masks = masks.type(torch.LongTensor) if not self.binary else masks.float()
                if self.model.out_ch > 1:
                    masks = masks.squeeze(1)
                masks = masks.to(self.device)
                imgs.append(images)
                targets.append(masks)
                prediction_idx.append(idx)

            prediction_idx = torch.cat(prediction_idx)
            imgs = torch.vstack(imgs)
            targets = torch.vstack(targets)
            return (imgs, targets, prediction_idx)

    def predict(self, images):
        predictions = self.model(images, features=False)
        return predictions

    def optimize_prior(self, n_steps=100):
        self.opt_prior = optimize_prior(
            prior_prec=self.prior, W=self.W, device=self.device, n_steps=n_steps
        ).to(device=self.device)

    def get_covariance(self):
        if (
            self.Train_Image == None
            and self.Train_Target == None
            and self.Train_Predictions == None
        ):
            return ValueError
        Sigma = compute_covariance(
            prior=self.opt_prior,
            W=self.W,
            predictions=self.Train_Predictions,
            target=self.Train_Target,
            device=self.device,
            binary=self.binary,
        )
        self.Sigma = Sigma

    def prepare_laplace(self):
        self.Train_Images, self.Train_Target, self.Train_Predictions_idx = self.validate_data(
            self.Train_Data
        )
        self.Train_Predictions = self.predict(self.Train_Images)
        # Optimize prior precision
        self.optimize_prior()
        # Compute Hessian and get Variance-Covariance matrix
        self.get_covariance()

    def get_predictions(self, X_val):
        val_imgs, val_masks, prediction_idx = self.validate_data(X_val)
        predictions = self.LaplacePrediction.Predict(
            val_imgs,
            self.Sigma,
            model=self.model,
            W=self.W,
            bias=self.bias,
            device=self.device,
            validate_args=self.validate_args,
        )
        return (val_imgs, val_masks, predictions, prediction_idx)


# if __name__ == "__main__":
#     # Load MAP Model
#     import torch
#     from src.models.model import UNET, init_weights
#     from src.data.dataloader import train_split
#     import random
#     random.seed(0)
#     torch.manual_seed(0)  # Set Torch Seed
#     torch.cuda.manual_seed_all(0)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     model = UNET(in_ch=1, out_ch=1, bilinear_method=False, momentum=0.9)
#     model.load_state_dict(torch.load("models/MAP/PhC-C2DH-U373_100%_21_5_False.pth"))
#     model.to("cpu")
#     model.eval()
#     dataset = "PhC-C2DH-U373"
#     device = "cpu"
#     # Get Data
#     train_loader, val_loader, _, _, _, _ = train_split(
#         train_size="100%", dataset="PhC-C2DH-U373", batch_size=4, to_binary=True, seed=21
#     )
#     # train_imgs = []
#     # train_targets = []
#     # for batch in train_loader:
#     #     img, mask, _ = batch
#     #     images, masks, _ = batch
#     #     images = images.unsqueeze(1) if dataset != "warwick" else images
#     #     images = images.to(device=device, dtype=torch.float32)
#     #     masks = masks.type(torch.LongTensor)
#     #     if model.out_ch > 1:
#     #         masks = masks.squeeze(1)
#     #     masks = masks.to(device)
#     #     train_imgs.append(images)
#     #     train_targets.append(masks)
#     # train_imgs = torch.vstack(train_imgs)
#     # train_targets = torch.vstack(train_targets)

#     # val_imgs = []
#     # val_targets = []
#     # for batch in val_loader:
#     #     img, mask, _ = batch
#     #     images, masks, _ = batch
#     #     images = images.unsqueeze(1) if dataset != "warwick" else images
#     #     images = images.to(device=device, dtype=torch.float32)
#     #     masks = masks.type(torch.LongTensor)
#     #     if model.out_ch > 1:
#     #         masks = masks.squeeze(1)
#     #     masks = masks.to(device)
#     #     val_imgs.append(images)
#     #     val_targets.append(masks)
#     # val_imgs = torch.vstack(val_imgs)
#     # val_targets = torch.vstack(val_targets)
#     # print(train_targets.shape)
#     # print(val_targets.shape)

#     #train_predictions = model(train_imgs, features=False)
#     LaplaceFitter = Laplace(
#         model=model,
#         binary=True,
#         Train_Data = train_loader,
#         hessian_method=None,
#         method="MC",
#         n_samples=50,
#         prior=torch.tensor(1),
#         validate_args=True,
#         device=device
#     )
#     LaplaceFitter.prepare_laplace()

#     _,_,predictions,_ = LaplaceFitter.get_predictions(val_loader)
#     print(predictions[0,0:5,0,0,0])
#     i=0
#     for batch in val_loader:
#         images,masks,pred_idx = batch
#         images = images.unsqueeze(1) if dataset != "warwick" else images
#         images = images.to(device=device, dtype=torch.float32)
#         masks = masks
#         if model.out_ch > 1:
#             masks = masks.squeeze(1)
#         masks = masks.to(device)
#         if i>0:
#             break
#         _,_,pred_i,_ = LaplaceFitter.get_predictions({'images':images,'masks':masks,'idx':pred_idx})
#         i+=1
#     print(pred_i[0,0:5,0,0,0])


#     from src.visualization.viz_tools import viz_batch

#     ged = viz_batch(
#         images=val_imgs[0:4],
#         masks=val_targets[0:4],
#         predictions=predictions[0:4],
#         cols=["img", "mask", "pred", "var"],
#         from_logits=True,
#         reduction=True,
#         save_=False,
#         dataset_type="membrane",
#         dataset="PhC-C2DH-U373",
#         save_path=None,
#     )

####

# from src.models.model import UNET,init_weights
# from src.data.dataloader import train_split
# model = UNET(in_ch=3,out_ch=1,bilinear_method=False,momentum=0.9)
# model.load_state_dict(torch.load('models/MAP/warwick_100%_21_5_False.pth'))
# model.to('cpu')
# model.eval()
# dataset='warwick'
# device = 'cpu'
# # Get Data
# train_loader, val_loader, _, _, _, _ = train_split(train_size='100%',dataset='warwick',batch_size=4,to_binary=True,seed=21)
# train_imgs = []
# train_targets = []
# for batch in train_loader:
#     img,mask,_ = batch
#     images, masks, _ = batch
#     images = images.unsqueeze(1) if dataset != "warwick" else images
#     images = images.to(device=device, dtype=torch.float32)
#     masks = masks.type(torch.LongTensor)
#     if model.out_ch > 1:
#         masks = masks.squeeze(1)
#     masks = masks.to(device)
#     train_imgs.append(images)
#     train_targets.append(masks)
# train_imgs=torch.vstack(train_imgs)
# train_targets=torch.vstack(train_targets)

# val_imgs = []
# val_targets = []
# for batch in val_loader:
#     img,mask,_ = batch
#     images, masks, _ = batch
#     images = images.unsqueeze(1) if dataset != "warwick" else images
#     images = images.to(device=device, dtype=torch.float32)
#     masks = masks.type(torch.LongTensor)
#     if model.out_ch > 1:
#         masks = masks.squeeze(1)
#     masks = masks.to(device)
#     val_imgs.append(images)
#     val_targets.append(masks)
# val_imgs=torch.vstack(val_imgs)
# val_targets=torch.vstack(val_targets)
# print(train_targets.shape)
# print(val_targets.shape)

# model = UNET(in_ch=3,out_ch=20,bilinear_method=False)
# images = torch.rand((128,3,64,64))
# targets = torch.rand((128,64,64)).round().long()
# W = list(model.parameters())[-2]
# train_predictions = model(images,features=False)
# model = extend(model)
# loss_func = extend(torch.nn.CrossEntropyLoss(reduction='sum'))
# loss = loss_func(train_predictions,targets)
# with backpack(extensions.KFAC()):
#     loss.backward()
# ged=2

# W = list(model.parameters())[-2]
# train_predictions = model(train_imgs,features=False)
# extend(model.out)
# loss_func = extend(torch.nn.BCEWithLogitsLoss(reduction='sum'))
# loss = loss_func(train_predictions.squeeze(1),train_targets.float())
# with backpack(extensions.KFAC()):
#     loss.backward()
# from src.models.laplace_utils import KFLA
# model_kfla = KFLA(model)
# model_kfla.get_hessian(train_loader,binary=True)
# ged=2

# train_predictions = model(train_imgs,features=False)
# LaplaceFitter = Laplace(model=model,binary=True,Train_Predictions=train_predictions,Train_Target=train_targets.float(),hessian_method=None,method='Probit',n_samples=1000,prior=1,validate_args=True)
# LaplaceFitter.get_predictions(val_imgs[0:50])

# from src.models.model_lin import UNET,init_weights

# #model = UNET(in_ch=3,out_ch=2,linear_last_layer=True,conv_weight_init=False)
# import torch.nn as nn
# import numpy as np
# m, n = 0,2
# h = 16  # num. hidden units
# k = 4  # num. classes

# # class ToyModel(torch.nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.start = torch.nn.Conv2d(3,64,kernel_size=1)
# #         self.out = torch.nn.Linear(64*64*64,2,bias=False)
# #     def forward(self,x):
# #         y = self.start(x)
# #         y = torch.flatten(y,1)
# #         return self.out(y)
# class ToyModel(nn.Module):

#     def __init__(self):
#         super(ToyModel, self).__init__()

#         self.feature_extr = nn.Sequential(
#             nn.Linear(n, h),
#             nn.BatchNorm1d(h),
#             nn.ReLU(),
#             nn.Linear(h, h),
#             nn.BatchNorm1d(h),
#             nn.ReLU()
#         )

#         self.clf = nn.Linear(h, k, bias=False)

#     def forward(self, x):
#         x = self.feature_extr(x)
#         return self.clf(x)
# model = ToyModel()
# #images = torch.rand((16,3,64,64))
# #targets = torch.rand((16,64,64)).long()
# #images = torch.rand((16,3,64,64))
# #targets = torch.rand((16)).round().long()
# from sklearn import datasets
# X,Y = datasets.make_blobs(n_samples=500,centers=4,cluster_std=1.2,center_box=(4,7.5),random_state=37)
# images,targets = torch.from_numpy(X).float(),torch.from_numpy(Y).long()
# opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# for it in range(5000):
#     y = model(images)
#     l = F.cross_entropy(y, targets)
#     l.backward()
#     opt.step()
#     opt.zero_grad()

# model.eval()
# W = list(model.parameters())[-1]
# print(W)
# train_predictions = model(images)
# extend(model.clf)
# loss_func = extend(torch.nn.CrossEntropyLoss(reduction='sum'))
# loss = loss_func(train_predictions,targets)

# with backpack(extensions.KFAC()):
#     loss.backward()
# A,B=W.kfac
