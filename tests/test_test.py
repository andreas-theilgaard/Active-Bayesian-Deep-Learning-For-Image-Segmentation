# from src.config import find_best_device,Config
# import torch
# import torchvision
# from torch.utils.data import DataLoader
# import os
# from PIL import Image
# from torch.utils.data import Dataset
# import numpy as np
# from torchvision import transforms
# def unwrap_batch(batch):
#     images,masks = batch
#     masks = masks.squeeze(1) if len(masks.shape) >4 else masks #[batch_size,width,height,number_classes]
#     masks = masks.permute(0,3,1,2).type(torch.float32 ) # [batch_size,number_classes,width,height]
#     return images,masks

# TRAIN_IMG_DIR = "/Users/andreastheilgaard/Desktop/U_net_exp/data/train_images/"
# TRAIN_MASK_DIR = "/Users/andreastheilgaard/Desktop/U_net_exp/data/train_masks/"
# VAL_IMG_DIR = "/Users/andreastheilgaard/Desktop/U_net_exp/data/val_images/"
# VAL_MASK_DIR = "/Users/andreastheilgaard/Desktop/U_net_exp/data/val_masks/"


# import torch
# import torch.nn as nn
# import torchvision.transforms.functional as TF
# import torchvision

# #ged = torch.rand(2,3,64,64)
# #ged=torch.load('test.pt')

# ##############################
# # U-Net Building Blocks:
# # - Encoder
# #    - Conv Block
# # - Decoder
# # - Bottleneck
# # ##############################

# class Conv_Block(nn.Module):
#     def __init__(self, in_ch : int, out_ch : int, dropout_rate: float, enable_dropout : bool, act : torch.nn.Module = nn.ReLU(inplace=True)):
#         super(Conv_Block,self).__init__()
#         self.layers = []
#         self.act = act
#         self.enable_dropout = enable_dropout

#         self.layers.append(nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride=1, padding=1, bias=False))
#         if self.enable_dropout:
#             self.layers.append(nn.Dropout(p=dropout_rate))
#         self.layers.append(nn.ReLU(inplace=True))
#         self.layers.append(nn.BatchNorm2d(out_ch))

#         self.layers.append(nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=1, bias=False))
#         if self.enable_dropout:
#             self.layers.append(nn.Dropout(p=dropout_rate))
#         self.layers.append(nn.ReLU(inplace=True))
#         self.layers.append(nn.BatchNorm2d(out_ch))
#         self.layers = nn.Sequential(*self.layers)

#     def forward(self, x):
#         return self.layers(x)

# class Encoder(nn.Module):
#     """
#     The encoder consist of convolution block
#     layers followed by max pooling.
#     """
#     def __init__(self,encoder_channels : list,dropout_rate:float,enable_dropout:bool,dropout_pool_rate: float, enable_pool_dropout : bool):
#         super(Encoder,self).__init__()
#         self.encoder_blocks = nn.ModuleList([Conv_Block(encoder_channels[x],encoder_channels[x+1],dropout_rate,enable_dropout) for x in range(len(encoder_channels)-1)])
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.enable_pool_dropout = enable_pool_dropout
#         self.dropout = nn.Dropout(p=dropout_pool_rate) # define dropout

#         layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
#         if self.enable_pool_dropout:
#             layers.append(nn.Dropout(p=dropout_pool_rate))
#         self.layers = nn.Sequential(*layers)


#     def forward(self,x):
#         forward_encoder_blocks = []
#         for encoder_block in self.encoder_blocks:
#             x = encoder_block(x)
#             forward_encoder_blocks.append(x) # skip connections we will save for use in decoder phase
#             x = self.pool(x)
#             x = self.dropout(x) if self.enable_pool_dropout else x
#         return forward_encoder_blocks

# # class Encoder(nn.Module):
# #     def __init__(self,in_channel,out_channel,dropout_rate,enable_dropout,dropout_pool_rate,enable_pool_dropout):
# #         super(Encoder,self).__init__()

# #         self.layers = []
# #         self.layers.append(Conv_Block(in_channel,out_channel,dropout_rate,enable_dropout))
# #         self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
# #         if enable_pool_dropout:
# #             self.layers.append(nn.Dropout(p=dropout_pool_rate))
# #         self.out = nn.Sequential(*self.layers)

# #     def forward(self,x):
# #         return self.out(x)


# class Decoder(nn.Module):
#     """
#     The decoder will be used to upsample
#     the features
#     """
#     def __init__(self,decoder_channels:list,dropout_rate:float,enable_dropout:bool):
#         super(Decoder,self).__init__()
#         self.decoder_channels = decoder_channels
#         self.up_sample = nn.ModuleList([nn.ConvTranspose2d(decoder_channels[x], decoder_channels[x+1], kernel_size=2, stride=2, padding=0) for x in range(len(decoder_channels)-1)])
#         self.conv = nn.ModuleList([Conv_Block(decoder_channels[x], decoder_channels[x+1],dropout_rate,enable_dropout) for x in range(len(decoder_channels)-1)])

#     def forward(self,x,skip_connections):
#         for i in range(len(self.decoder_channels)-1):
#             x = self.up_sample[i](x)
#             skip_conn = skip_connections[i]
#             if x.shape != skip_conn.shape:
#                 x = TF.resize(x,size=skip_conn.shape[2:])
#                 #skip_conn = self.resize_by_crop(skip_conn,x)
#             x = torch.cat((x, skip_conn), axis=1) # concat skip_conn and decoder layer
#             #x = torch.cat([skip_conn,x],dim=1)
#             x = self.conv[i](x)
#         return x

#     def resize_by_crop(self,skip_conn,x):
#         bs,c,img_height,img_width = x.shape
#         return torchvision.transforms.CenterCrop([img_height,img_width])(skip_conn)


# class UNET(nn.Module):
#     def __init__(self,n_classes :int = 50, enc_channels : list = [3,64,128,256,512,1024],
#             dec_channels : list = [1024,512,256,128,64], dropout_rate : float = 0.5, enable_dropout : bool = False,
#             dropout_pool_rate : float = 0.5, enable_pool_dropout : bool = False
#             ):
#         super(UNET,self).__init__()

#         # Define variables
#         self.n_classes = n_classes
#         self.dropout_rate = dropout_rate
#         self.enable_dropout = enable_dropout
#         self.dropout_pool_rate = dropout_pool_rate
#         self.enable_pool_dropout = enable_pool_dropout

#         # Encoder Layers
#         self.Encoder = Encoder(enc_channels,dropout_rate,enable_dropout,dropout_pool_rate, enable_pool_dropout)

#         # Decoder Layers
#         self.Decoder = Decoder(dec_channels,dropout_rate,enable_dropout)

#         # Final Classification head
#         self.outputs = nn.Conv2d(dec_channels[-1], self.n_classes, kernel_size=1, padding=0)

#     def forward(self,x):
#         encode_layers = self.Encoder(x)
#         bottleneck = encode_layers[::-1][0]
#         skip_connections = encode_layers[::-1][1:]
#         decode_layers = self.Decoder(bottleneck,skip_connections)
#         outputs = self.outputs(decode_layers)
#         return outputs

# # device = find_best_device()
# # model = UNET(n_classes=2).to(device)
# # out = model(ged.to(device))

# # out[0][0][0][0]

# import torch
# import torch.nn as nn
# import torchvision.transforms.functional as TF


# #Define Double Conv class that will be used throughout the U-net archectiecture

# # class DoubleConv(nn.Module):
# #     def __init__(self, in_channels,out_channels):
# #         super(DoubleConv,self).__init__()
# #         self.conv = nn.Sequential(
# #             nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False), # same convolution
# #             nn.BatchNorm2d(out_channels), # note in U-net paper batchnorm is not used as introduced in 2016 and paper from 2016, but Saidu paper uses batchnorm as well
# #             nn.ReLU(inplace=True),
# #             # Now we do the same procedure for the second "same concolution 'block' "
# #             nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False), # same convolution
# #             nn.BatchNorm2d(out_channels), # note in U-net paper batchnorm is not used as introduced in 2016 and paper from 2016, but Saidu paper uses batchnorm as well
# #             nn.ReLU(inplace=True)
# #         )

# #     def forward(self,x):
# #         return self.conv(x)


# # # Now let's define the U-net archectechture class, since we have our helper class implemented


# # # be aware of out_channels in paper =2
# # class UNET(nn.Module):
# #     def __init__(self,out_channels=2,in_channels=3, features=[64,128,256,512]):
# #         super(UNET,self).__init__()
# #         self.ups = nn.ModuleList()
# #         self.downs = nn.ModuleList()
# #         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

# #         # Down Part of Unet
# #         for feature in features:
# #             self.downs.append(DoubleConv(in_channels, feature))
# #             in_channels = feature

# #         # Up part of Unet
# #         for feature in features[::-1]:
# #             self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
# #             self.ups.append(DoubleConv(feature*2,feature))

# #         # Bottom layers
# #         self.bottleneck = DoubleConv(features[-1],features[-1]*2)
# #         self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)


# #     def forward(self,x):
# #         skip_connections = []
# #         for down in self.downs:
# #             x = down(x)
# #             skip_connections.append(x)
# #             x = self.pool(x)

# #         # Now bottleneck

# #         x = self.bottleneck(x)
# #         skip_connections = skip_connections[::-1]
# #         for idx in range(0,len(self.ups),2):
# #             x = self.ups[idx](x)
# #             skip_connection = skip_connections[idx//2]

# #             if x.shape != skip_connection.shape:
# #                 x = TF.resize(x,size=skip_connection.shape[2:])

# #             #concat_skip = torch.cat((skip_connection,x),dim=1)
# #             concat_skip = torch.cat((x,skip_connection),dim=1)
# #             x = self.ups[idx+1](concat_skip)
# #         out = self.final_conv(x)
# #         return out

# class CarvanaDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = [x for x in os.listdir(image_dir) if x != '.DS_Store']
#         self.transform = transforms.Compose([
#             transforms.Resize((64,64)),
#             transforms.PILToTensor()])

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.image_dir, self.images[index])
#         mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
#         #image = np.array(Image.open(img_path).convert("RGB"))
#         #mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
#         image = Image.open(img_path).convert('RGB')
#         image = self.transform(image)
#         mask = Image.open(mask_path) #.convert('RGB') # maybe to gray
#         mask = self.transform(mask)

#         #mask[mask == 255.0] = 1.0
#         mask = mask.type(torch.uint8)
#         #masks = tf.keras.utils.to_categorical(mask,self.n_classes) #require tensorflow
#         masks = np.eye(2,dtype='uint8')[mask]
#         masks = torch.from_numpy(masks)
#         return image, masks


# def get_loaders(
#     train_dir,
#     train_maskdir,
#     val_dir,
#     val_maskdir,
#     batch_size,
#     num_workers=4,
#     pin_memory=True,
# ):
#     train_ds = CarvanaDataset(
#         image_dir=train_dir,
#         mask_dir=train_maskdir,
#     )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=True,
#     )

#     val_ds = CarvanaDataset(
#         image_dir=val_dir,
#         mask_dir=val_maskdir,
#     )

#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=False,
#     )

#     return train_loader, val_loader

# import torch
# import torch.nn as nn
# from src.config import find_best_device,Config
# #from src.models.Unet_model import UNET
# from tqdm import tqdm
# import argparse
# import torch.optim as optim
# #from src.data.dataloader import get_loaders,unwrap_batch
# import numpy as np
# from torch.utils.data import DataLoader
# from src.models.model_utils import DICE_Batch_Score
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# LEARNING_RATE = 1e-4
# DEVICE = "mps"
# BATCH_SIZE = 16
# NUM_EPOCHS = 5
# NUM_WORKERS = 2
# IMAGE_HEIGHT = 128  # 1280 originally
# IMAGE_WIDTH = 128  # 1918 originally
# PIN_MEMORY = True
# LOAD_MODEL = False

# def train():
#     parser = argparse.ArgumentParser(description="Training arguments")
#     parser.add_argument("--lr", default=1e-4)
#     parser.add_argument("--bs", default=5)
#     parser.add_argument("--workers", default=4)
#     parser.add_argument("--epochs", default=5)
#     parser.add_argument('--enable_dropout',default=True)
#     parser.add_argument('--enable_pool_dropout',default=True)
#     parser.add_argument('--dropout_rate',default=0.5)
#     parser.add_argument('--dropout_pool_rate',default=0.5)
#     args = parser.parse_args()
#     print(args)

#     # Get Data
#     # train_transform = transforms.Compose([
#     #         transforms.Resize((64,64)),
#     #         transforms.Normalize((0., 0., 0.), (1., 1., 1.)),
#     #         transforms.PILToTensor()])

#     # val_transforms = transforms.Compose([
#     #         transforms.Resize((64,64)),
#     #         transforms.Normalize((0., 0., 0.), (1., 1., 1.)),
#     #         transforms.PILToTensor()])
#     # train_transform = A.Compose(
#     #     [
#     #         A.Resize(height=64, width=64),
#     #         A.Rotate(limit=35, p=1.0),
#     #         A.HorizontalFlip(p=0.5),
#     #         A.VerticalFlip(p=0.1),
#     #         A.Normalize(
#     #             mean=[0.0, 0.0, 0.0],
#     #             std=[1.0, 1.0, 1.0],
#     #             max_pixel_value=255.0,
#     #         ),
#     #         ToTensorV2(),
#     #     ],
#     # )

#     # val_transforms = A.Compose(
#     #     [
#     #         A.Resize(height=64, width=64),
#     #         A.Normalize(
#     #             mean=[0.0, 0.0, 0.0],
#     #             std=[1.0, 1.0, 1.0],
#     #             max_pixel_value=255.0,
#     #         ),
#     #         ToTensorV2(),
#     #     ],
#     # )

#     device = find_best_device()
#     #model = UNET(out_channels=2).to(DEVICE)
#     model = UNET(n_classes=2).to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     train_loader, val_loader = get_loaders(
#         TRAIN_IMG_DIR,
#         TRAIN_MASK_DIR,
#         VAL_IMG_DIR,
#         VAL_MASK_DIR,
#         BATCH_SIZE,
#         NUM_WORKERS,
#         PIN_MEMORY,
#     )


#     # Initilize model
#     # device = find_best_device() # get best device
#     # model = UNET(n_classes= Config.n_classes[args.dataset], enc_channels = [3,64,128,256,512,1024],
#     #         dec_channels = [1024,512,256,128,64], dropout_rate = args.dropout_rate, enable_dropout = args.enable_dropout,
#     #         dropout_pool_rate = args.dropout_pool_rate, enable_pool_dropout = args.enable_pool_dropout)
#     # model.to(device)
#     # # intitliaze model params
#     # criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     data = tqdm(range(args.epochs),desc='Model Training')

#     for epoch in data:
#         train_loss_tracker = torch.tensor([]).to(device)
#         train_dice_tracker = torch.tensor([]).to(device)


#         for batch in tqdm(train_loader):
#             images,masks = unwrap_batch(batch)
#             images,masks = images.to(device),masks.to(device)
#             #images = images.permute(0,3,1,2)
#             images = images.to(torch.float32)

#             #masks=masks.float().unsqueeze(1)

#             #Zero Gradients Before Optimize Step
#             optimizer.zero_grad()

#             # Forward Pass
#             outputs = model(images)
#             loss = criterion(outputs,masks)
#             loss.backward()
#             optimizer.step()

#             dice_score = DICE_Batch_Score(masks,outputs)

#             train_loss_tracker = torch.cat((train_loss_tracker,torch.tensor([loss]).to(device)))
#             train_dice_tracker = torch.cat((train_dice_tracker,torch.tensor([dice_score.item()]).to(device)))

#         # # # Validation loop
#         val_loss_tracker = torch.tensor([]).to(device)
#         val_dice_tracker = torch.tensor([]).to(device)
#         model.eval()
#         with torch.no_grad():
#             for batch in tqdm(val_loader):
#                 images,masks = unwrap_batch(batch)
#                 images,masks = images.to(device),masks.to(device)
#                 images = images.to(torch.float32)

#                 outputs = model(images)
#                 loss = criterion(outputs, masks)

#                 dice_score = DICE_Batch_Score(masks,outputs)

#                 val_loss_tracker = torch.cat((val_loss_tracker,torch.tensor([loss]).to(device)))
#                 val_dice_tracker = torch.cat((val_dice_tracker,torch.tensor([dice_score.item()]).to(device)))


#         # put model back to train
#         model.train()


#         data.set_postfix({"Train Loss": train_loss_tracker.mean().item(),
#         "Train DICE":train_dice_tracker.mean().item(),
#         "Val Loss":val_loss_tracker.mean().item(),
#         "Val DICE":val_dice_tracker.mean().item(),
#         })


#     # do train loop

# if __name__ == "__main__":
#     train()
