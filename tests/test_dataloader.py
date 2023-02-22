# from src.data.dataloader import get_loaders,unwrap_batch
# import torch
# import numpy as np
# from src.config import Config
# import os


# def test_datashapes_images():
#     all_datasets = list(Config.n_classes.keys())
#     for dataset in all_datasets:
#         data = get_loaders(dataset_type = dataset,batch_size = 1,img_height = 64, img_width=64,num_workers=4,test='image')
#         for batch in data:
#             print(batch.shape)
#             assert batch.shape[1] == 3


# def test_datashapes_masks():
#     all_datasets = list(Config.n_classes.keys())
#     for dataset in all_datasets:
#         data = get_loaders(dataset_type = dataset,batch_size = 1,img_height = 64, img_width=64,num_workers=4,test='mask')
#         for batch in data:
#             print(batch.shape,dataset)
#             assert batch.shape[1] == 1


# def test_equal_sizes():
#     all_datasets = list(Config.n_classes.keys())
#     for dataset in all_datasets:
#         data_path = f"data/raw/{dataset}/dataset"
#         images  = [x for x in os.listdir(f"{data_path}/image") if x != '.DS_Store']
#         masks = [x for x in os.listdir(f"{data_path}/label") if x != '.DS_Store']
#         for i in range(len(images)):
#             assert masks[i] == images[i]


# def test_train_val_split(training_ratios = np.arange(.99, .01, -.10),batch_sizes=[4,32]):
#     for dataset in list(Config.n_classes.keys()):
#         for batch_size in batch_sizes:
#             data = get_loaders(dataset_type=dataset,batch_size = batch_size)
#             for ratio in training_ratios:
#                 train_data, val_data = torch.utils.data.random_split(data.dataset,
#                 lengths=[1-ratio,ratio],
#                 generator=torch.Generator().manual_seed(np.random.randint(10)))
#                 print(len(data.dataset))
#                 assert len(data.dataset) == len(train_data)+len(val_data)

# def test_dataloader_shapes():
#     all_datasets = list(Config.n_classes.keys())
#     for dataset in all_datasets:
#         for batch_size in [64]:
#             for img_size in [64]:
#                 data = get_loaders(dataset_type=dataset,batch_size=batch_size,img_height=img_size, img_width=img_size)
#                 batch = next(iter(data))
#                 images,masks = unwrap_batch(batch)
#                 assert list(images.shape) == [batch_size,3,img_size, img_size]
#                 assert list(masks.shape) == [batch_size,Config.n_classes[dataset],img_size, img_size]
