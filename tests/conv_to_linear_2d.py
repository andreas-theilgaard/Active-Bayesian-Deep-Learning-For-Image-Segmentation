import torch
import torch.nn as nn


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d]:
        nn.init.xavier_normal_(model.weight)
        nn.init.normal_(model.bias, std=0.001)


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=True))

    def forward(self, x):
        return self.layers(x)


class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, weight_init=None, bias=None):
        super().__init__()
        one_by_one_conv = nn.Linear(input_dim, output_dim, bias=True)
        if torch.is_tensor(weight_init):
            with torch.no_grad():
                one_by_one_conv.weight.copy_(weight_init.view(one_by_one_conv.weight.shape))
                one_by_one_conv.bias.copy_(bias.view(one_by_one_conv.bias.shape))
        self.layers = nn.Sequential(one_by_one_conv)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    torch.manual_seed(17)
    # input_dims = (1,10,3,3)
    input_dims = (10, 3, 3)
    output_dim = 1

    conv = CNN(input_dims[0], output_dim)

    # A = torch.rand(input_dims[0],input_dims[1],input_dims[2],input_dims[3])
    A = torch.rand(input_dims[0], input_dims[1], input_dims[2])

    # conv = CNN(input_dims[1],output_dim)
    conv.apply(init_weights)

    for name, param in conv.named_parameters():
        if name == "layers.0.weight":
            weight_param = param
        elif name == "layers.0.bias":
            bias_param = param
    lin = FNN(input_dims[0], output_dim, weight_param, bias_param)
    # lin= FNN(input_dims[1],output_dim,weight_param,bias_param)
    # lin.apply(init_weights)
    # conv.apply(init_weights)

    # conv = nn.Conv2d(input_dims[1], output_dim, kernel_size=1, bias=False) #in_dim=3, out_dim=5
    # lin = nn.Linear(input_dims[1],output_dim,bias=False)
    # with torch.no_grad():
    #    lin.weight.copy_(conv.weight.view(lin.weight.shape))
    # lin_out = lin(A.permute(0,3,2,1)).permute(0,3,2,1)
    lin_out = lin(A.permute(2, 1, 0)).permute(2, 1, 0)
    conv_out = conv(A)
    print(lin_out, lin_out.shape)
    print(conv_out, conv_out.shape)
