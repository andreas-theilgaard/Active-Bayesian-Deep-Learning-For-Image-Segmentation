import torch
import torch.nn as nn

# A = torch.rand(1,2,2)
# A.transpose(1,2)

# A.permute(0,2,1)


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d]:
        nn.init.xavier_normal_(model.weight)
        # nn.init.normal_(model.bias, std=0.001)


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

    def forward(self, x):
        return self.layers(x)


class FNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False))

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    torch.manual_seed(17)
    input_dims = (1, 10, 3)
    output_dim = 1

    A = torch.rand(input_dims[0], input_dims[1], input_dims[2])
    # conv = nn.Conv1d(input_dims[1], output_dim, kernel_size=1, bias=False) #in_dim=3, out_dim=5
    # lin = nn.Linear(input_dims[1],output_dim,bias=False)
    lin = FNN(input_dims[1], output_dim)
    lin.apply(init_weights)
    conv = CNN(input_dims[1], output_dim)
    conv.apply(init_weights)
    # with torch.no_grad():
    #    lin.weight.copy_(conv.weight.view(lin.weight.shape))
    lin_out = lin(A.permute(0, 2, 1)).permute(0, 2, 1)
    conv_out = conv(A)
    print(lin_out, lin_out.shape)
    print(conv_out, lin_out.shape)
