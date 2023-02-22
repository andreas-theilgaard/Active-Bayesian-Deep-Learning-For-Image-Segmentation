# from src.models.model import UNET
# import torch
# from src.data.dataloader import get_loaders,unwrap_batch
# from src.config import Config

# def test_model():
#     dataset_type = list(Config.n_classes.keys())
#     for data_type in dataset_type:
#         for img_shape in [64,128]:
#             data = get_loaders(dataset_type=data_type,batch_size=4,img_height = img_shape, img_width =img_shape)
#             batch = next(iter(data))
#             images,masks = batch#unwrap_batch(batch)

#             model = UNET(n_classes=Config.n_classes[data_type])
#             out = model(images)
#             assert out.shape==masks.shape

# if __name__ == "__main__":
#     test_model()
