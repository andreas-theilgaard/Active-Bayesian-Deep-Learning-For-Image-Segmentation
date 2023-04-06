# import torch
# import torch.nn as nn

# # Let's assume input for 1d conv is
# x = torch.tensor([[
#     [0.608502902,0.833821936,0.793526093],
#     [0.905739925,0.95717805,0.809504577],
#     [0.930305136,0.966781513,0.440928965]]])
# x.shape # [1,3,3] batch_size,outdimension, width

# conv = nn.Conv1d(3, 5, kernel_size=1, bias=False) #in_dim=3, out_dim=5
# conv.weight.shape # out_dim,in_dim,kernel_size


# # so after applying the convultion, we now we should end up with something like this
# # 1,5,3
# conv(x).shape

# # Now Assume weights are given by
# # _conv_weight = torch.tensor([[0.484249785, 0.419076606, 0.108487291]])
# # conv_weight = torch.cat(5*[_conv_weight])
# # conv_weight = conv_weight.view(conv.weight.shape)

# # with torch.no_grad():
# #     conv.weight.copy_(conv_weight)
# #### now let's get the out
# conv(x)
# torch.sum(conv.weight*x,dim=1)

# # okay so this is the interesting part
# lin = nn.Linear(3,5,bias=False) # input, output_dim
# lin.weight.shape # (5,3)
# # So we know we should have output dim of 1,5,3 = batch_size,output_dim, img_shape
# # we know input to linear must be 3,3
# x

# # So this one is just making a regular transpose
# (x.transpose(-1, -2)).transpose(-1,2)

# lin(x.transpose(-1, -2)).transpose(-1, -2).shape
# # as desired


# # now let's make the exact same output
# with torch.no_grad():
#     lin.weight.copy_(conv.weight.view(lin.weight.shape))

# out_1 = lin(x.transpose(1,2)).transpose(1,2)
# out_1

# conv(x)


# import torch
# torch.manual_seed(17)
# A = torch.rand(1,10,3,3)
# conv_layer = torch.nn.Conv2d(10,1,kernel_size=1,bias=False)


# in_channels, out_channels = conv_layer.weight.shape[:2]
# height, width = A.shape[-2:]
# Correct_Shape = in_channels * height * width

# lin_layer = torch.nn.Linear(Correct_Shape,Correct_Shape,bias=False)

# # Apply conv_layer.weight to lin_layer
# in_channels, out_channels = conv_layer.weight.shape[:2]
# height, width = A.shape[-2:]
# Correct_Shape = in_channels * height * width

# lin_layer.weight.data = conv_layer.weight.view(out_channels, Correct_Shape)

# A = torch.rand(1,10,3,3)

# # Make forward pass with conv_layer and lin_layer on A
# out_conv = conv_layer(A)
# out_lin = lin_layer(A.view(1, Correct_Shape)).view(1, 1, height, width)

# print(out_conv)
# print(out_lin)


# import torch
# torch.manual_seed(17)
# conv_layer = torch.nn.Conv2d(10,1,kernel_size=1,bias=False)
# lin_layer = torch.nn.Linear(Correct_Shape,Correct_Shape,bias=False)

# # Apply conv_layer.weight to lin_layer
# A = torch.rand(1,10,3,3)
# # Make forward pass with conv_layer and lin_layer on A

# torch.matmul(lin_layer.weight,(A.reshape(1*10*3*3,-1))).reshape(1,1,3,3)


# import torch.nn as nn
# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(nn.Conv2d(10, 1, kernel_size=1, bias=False))

#     def forward(self, x):
#         return self.layers(x)

# class FNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(10 * 3 * 3, 3 * 3,bias=False)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = x.view(x.size(0), 1, 3, 3)
#         return x

# def init_weights(model):
#     if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d]:
#         nn.init.xavier_normal_(model.weight)
#         if model.bias is not None:
#             nn.init.normal_(model.bias, std=0.001)
#     elif type(model) == nn.BatchNorm2d:
#         nn.init.constant_(model.weight, 1)
#         nn.init.constant_(model.bias, 0)

#         if model.running_mean is not None:
#             nn.init.constant_(model.running_mean, 0)
#             nn.init.constant_(model.running_var, 1)

#         if model.num_batches_tracked is not None:
#             nn.init.constant_(model.num_batches_tracked, 0)

# torch.manual_seed(17)

# A = torch.rand(1, 10, 3, 3)
# ModelA = CNN()
# ModelA.apply(init_weights)
# ModelB = FNN()
# ModelB.apply(init_weights)

# # Set ModelB to use same weights as ModelA
# for name, param in ModelA.named_parameters():
#     if name == 'layers.0.weight':
#         weight_param = param
#         break

# with torch.no_grad():
#     ModelB.fc.weight.copy_(weight_param.view(9, -1))

# outA = ModelA(A)
# outB = ModelB(A)
# print((outA == outB).all())


# import torch.nn as nn

# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(nn.Conv2d(10, 1, kernel_size=1, bias=False))

#     def forward(self, x):
#         return self.layers(x)

# class FNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(10 * 3 * 3, 3 * 3,bias=False)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = x.view(x.size(0), 1, 3, 3)
#         return x

# def init_weights(model):
#     if type(model) in [nn.Conv2d, nn.ConvTranspose2d,nn.Linear,nn.Conv1d]:
#         nn.init.xavier_normal_(model.weight)
#         #nn.init.normal_(model.bias, std=0.001)

# import torch
# torch.manual_seed(17)

# A = torch.rand(1, 10, 3, 3)
# ModelA = CNN()
# ModelA.apply(init_weights)
# ModelB = FNN()
# ModelB.apply(init_weights)

# for name,param in ModelA.named_parameters():
#     if name == 'layers.0.weight':
#         weight_param=param

# # Set ModelB to use same weights as ModelA
# outA = ModelA(A)
# outB = ModelB(A)
# # Test if outA==outB


# torch.sum(weight_param*A,dim=1)
# ModelA(A)

# output_shape = (1,3,3)
# output_size = output_shape[-1] * output_shape[-2] * 1
# # Define the linear layer
# linear_layer = nn.Linear(in_features=10*3*3*1, out_features=output_size, bias=False)
# out_channels =1

# with torch.no_grad():
#     weight_tensor = weight_param.view(out_channels, -1)
#     linear_layer.weight.copy_(weight_tensor)
#     #if conv_layer.bias is not None:
#     #    linear_layer.bias.copy_(conv_layer.bias)

# linear_layer.weight.shape
# A.numel()

# linear_layer.weight

# linear_layer(A)

# GED = torch.rand(4,5)
# li = torch.nn.Linear(5,2,bias=False)
# li(GED)
# torch.matmul(GED,torch.transpose(li.weight,dim0=0,dim1=1))


# GED*

# li.weight.shape
# torch.transpose(li.weight,dim0=0,dim1=1)*GED

# GED*torch.rand(5,4)
# torch.matmul(GED,torch.rand(5,4))

# linear_layer(A)

# lin = nn.Linear(10, 1, bias=False)
# with torch.no_grad():
#     lin.weight.copy_(weight_param.view(lin.weight.shape))

# A.shape

# A.transpose(2,3).shape

# torch.transpose(A,dim0=2,dim1=3).shape
# weight_param.shape
# lin.weight.shape

# torch.transpose(A,dim0=2,dim1=3)*lin.weight


# lin(A.transpose(2,3))

# out_1 = lin(x.transpose(1,2))
# out_1


# for name,param in ModelB.named_parameters():
#     print(name,param)


# #ModelA.layers.0.weight

# outA = ModelA(A)
# outB = ModelB(A)
# assert torch.allclose(outA, outB, atol=1e-6)


# import torch
# import torch.nn as nn
# torch.manual_seed(17)
# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(torch.Conv2d(10,1,kernel_size=1,bias=False))
#     def forward(self,x):
#         return self.layers(x)
# class FNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = None
#     def forward(self):
#         pass

# A = torch.rand(1,10,3,3)
# ModelA = CNN()
# ModelB = FNN()
# outA = ModelA(A)
# outB = ModelB(A)
# assert outA == outB


# import torch
# import torch.nn as nn
# from functools import reduce

# import torch.nn as nn

# torch.manual_seed(17)
# A = torch.rand(1,10,3,3)
# conv_layer = nn.Conv2d(10,1,kernel_size=1,bias=False)
# conv_layer.weight.shape

# (conv_layer.weight*A).shape
# torch.sum((conv_layer.weight*A),dim=1)

# conv_layer(A)

# A.shape
# A.numel()
# lin_layer = torch.nn.Linear(10,1)
# lin_layer.weight.shape
# linear_layer.weight.copy_(weight_tensor)

# lin_layer

# conv_layer.weight.view(lin_layer.weight.shape)

# with torch.no_grad():
#     lin.weight.copy_(conv_weight.view(lin.weight.shape))


# def conv2linear(conv_layer):
#     """
#     Converts a Conv2d layer to a Linear layer

#     Arguments:
#     conv_layer -- a Conv2d layer

#     Returns:
#     linear_layer -- a Linear layer with weights and biases that replicate the behavior of conv_layer
#     """

#     # Get the input and output dimensions of the conv_layer
#     in_channels = conv_layer.in_channels
#     out_channels = conv_layer.out_channels
#     kernel_size = conv_layer.kernel_size
#     stride = conv_layer.stride
#     padding = conv_layer.padding

#     # Compute the output dimensions of the conv_layer
#     input_shape = (1, in_channels, conv_layer.input_size[0], conv_layer.input_size[1])
#     with torch.no_grad():
#         dummy_input = torch.randn(input_shape)
#         output_shape = conv_layer(dummy_input).shape
#     output_size = output_shape[-1] * output_shape[-2] * out_channels

#     # Define the linear layer
#     linear_layer = nn.Linear(in_features=input_shape[1]*kernel_size[0]*kernel_size[1], out_features=output_size, bias=True)

#     # Copy the weights and biases from the conv_layer to the linear_layer
#     with torch.no_grad():
#         weight_tensor = conv_layer.weight.view(out_channels, -1)
#         linear_layer.weight.copy_(weight_tensor)
#         if conv_layer.bias is not None:
#             linear_layer.bias.copy_(conv_layer.bias)

#     return linear_layer

# torch.manual_seed(17)
# A = torch.rand(2,64,64,64)
# conv_layer = nn.Conv2d(64,1,kernel_size=1)

# outA=conv_layer(A)
# outB= conv2linear(conv_layer)(A)


# ###############


# # Lets assume this is out input tensor to our Conv 1x1
# x = torch.tensor([[
#     [0.608502902,0.833821936,0.793526093],
#     [0.905739925,0.95717805,0.809504577],
#     [0.930305136,0.966781513,0.440928965]]])

# conv = nn.Conv1d(3, 5, 1, bias=False)

# # Assum weights are these
# _conv_weight = torch.tensor([[0.484249785, 0.419076606, 0.108487291]])
# conv_weight = torch.cat(5*[_conv_weight])
# assert reduce((lambda x, y: x * y), [x for x in conv_weight.shape])==reduce((lambda x, y: x * y), [x for x in conv.weight.shape])
# conv_weight = conv_weight.view(conv.weight.shape)


# with torch.no_grad():
#     conv.weight.copy_(conv_weight)

# conv(x)

# ## Now consider fully connected layer
# Linear_Layer = nn.Linear(3,5,bias=False)

# x.transpose(-1,-2)
# x


# Linear_Layer(x.transpose(-1, -2)).transpose(-1, -2).shape


# def init_weights(model):
#     if type(model) in [nn.Conv2d, nn.ConvTranspose2d,nn.Linear,nn.Conv1d]:
#         nn.init.xavier_normal_(model.weight)
#         nn.init.normal_(model.bias, std=0.001)

# class ConvLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(nn.Conv1d(64, 1, kernel_size=1, bias=True))
#     def forward(self,x):
#         return self.layers(x)

# class LinearLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(nn.Linear(64*64,64*1,bias=True))
#     def forward(self,x):
#         return self.layers(x)

# torch.manual_seed(17)
# A = torch.rand(1,64,64) # Batch_size, channels, L_out
# Conv_layer = ConvLayer()
# Conv_layer.apply(init_weights)
# Linear_Layer = LinearLayer()
# Linear_Layer.apply(init_weights)

# for name, param in Conv_layer.named_parameters():
#     if name == 'layers.0.bias':
#         print(name,param)

# for name, param in Linear_Layer.named_parameters():
#     if name == 'layers.0.bias':
#         print(name,param)


# out_conv = Conv_layer(A)


# if __name__ == "__main__":
#     torch.manual_seed(17)
#     A = torch.rand(1,64,64) # Batch_size, channels, L_out
#     Conv_layer = ConvLayer()
#     Conv_layer.apply(init_weights)
#     out_conv = Conv_layer(A)
#     #out_linear = torch.nn.Linear(64*64*64,64*64*1,bias=True)(A.reshape(64*64*64*1,))
#     #out_linear = out_linear.reshape(1,1,64,64)
#     #print(out_linear.shape)
#     print(out_conv.shape)
#     #print(out_conv[0][0][0][0],out_linear[0][0][0][0])


# # if __name__ == "__main__":
# #     torch.manual_seed(17)
# #     A = torch.rand(1,64,64,64)
# #     out_conv = torch.nn.Conv2d(64, 1, kernel_size=1, bias=True)(A)
# #     out_linear = torch.nn.Linear(64*64*64,64*64*1,bias=True)(A.reshape(64*64*64*1,))
# #     out_linear = out_linear.reshape(1,1,64,64)
# #     print(out_linear.shape)
# #     print(out_conv.shape)
# #     print(out_conv[0][0][0][0],out_linear[0][0][0][0])
