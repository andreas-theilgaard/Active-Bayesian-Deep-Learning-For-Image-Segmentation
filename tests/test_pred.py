# from src.data.dataloader import get_loaders
# from src.visualization.visualize import viz_batch,mask_to_class_channel
# import os
# from src.visualization.viz_utils import styles, viz_mask_helper
# import torch
# from src.config import Config
# import matplotlib.pyplot as plt
# import numpy as np

# base_path = 'predictions'
# dataset_type='PhC-C2DH-U373'
# images_path = [x for x in os.listdir(f"{base_path}/image") if x !='.DS_Store']
# mask_path = [x for x in os.listdir(f"{base_path}/mask") if x !='.DS_Store']
# pred_mask_path = [x for x in os.listdir(f"{base_path}/pred_mask") if x !='.DS_Store']

# sorted_img = np.argsort(np.array([float(x[:-4]) for x in images_path if x !='.DS_Store']))
# sorted_mask = np.argsort(np.array([float(x[:-4]) for x in mask_path if x !='.DS_Store']))
# sorted_pred = np.argsort(np.array([float(x[:-4]) for x in pred_mask_path if x !='.DS_Store']))

# def plot_prediction(img,mask,pred_mask,dataset_type):

#     fig, axes = plt.subplots(1, 3)
#     fig.set_size_inches(11,6)

#     # plot image
#     axes[0].imshow(img.permute(1,2,0))
#     axes[0].set_xlabel(f"Label")


#     # plot mask
#     mask = mask_to_class_channel(mask,Config.n_classes[dataset_type])
#     mask = viz_mask_helper(mask.squeeze(0),dataset_type)
#     axes[1].imshow(mask,cmap='viridis')
#     axes[1].set_xlabel(f"Target")


#     # plot mask
#     pred_mask = mask_to_class_channel(pred_mask.argmax(dim=0).unsqueeze(0),Config.n_classes[dataset_type])
#     pred_mask = viz_mask_helper(pred_mask.squeeze(0),dataset_type)
#     axes[2].imshow(pred_mask,cmap='viridis')
#     axes[2].set_xlabel(f"Prediction")

#     plt.show()

# for i in range(len(images_path)):
#     img = torch.load(f"{base_path}/image/{images_path[sorted_img[i]]}")
#     mask = torch.load(f"{base_path}/mask/{mask_path[sorted_mask[i]]}")
#     pred_mask = torch.load(f"{base_path}/pred_mask/{pred_mask_path[sorted_pred[i]]}")

#     plot_prediction(img,mask,pred_mask,dataset_type)
# pred_mask.shape
# mask.shape
# img.shape
# # masks = mask_to_class_channel(mask,Config.n_classes[dataset_type])

# processed_mask = viz_mask_helper(masks.squeeze(0),dataset_type)
# plt.imshow(processed_mask,cmap='viridis')
# plt.show()


# plt.imshow(img.permute(1,2,0))
# plt.show()


# mask_pred = mask_to_class_channel(pred_mask.argmax(dim=0).unsqueeze(0),Config.n_classes[dataset_type])
# processed_mask_pred = viz_mask_helper(mask_pred.squeeze(0),dataset_type)
# plt.imshow(processed_mask_pred,cmap='viridis')
# plt.show()


# plot_prediction(img,mask,pred_mask,dataset_type)
