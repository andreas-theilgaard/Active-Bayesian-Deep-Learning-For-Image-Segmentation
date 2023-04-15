import torch
from src.models.model import UNET, init_weights
import numpy as np
import random
from tqdm import tqdm
from src.models.laplace import Laplace


def enable_MCDropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def inference(
    models,
    model_params,
    data_loader,
    method,
    dataset,
    device,
    seed,
    torch_seeds,
    n_forward_passes=50,
    train_loader=None,
    binary=None,
):
    """
    models: list of models for inference
    data_loader: data_loader object
    method: which inference method to use
    """
    assert method in ["Normal", "MCD", "DeepEnsemble", "Laplace", "BatchNorm"]
    ensemble = []
    # go trhough each model in model list
    for i, model_path in enumerate(models):
        # set seeds
        random.seed(torch_seeds[i])
        np.random.seed(seed)
        torch.manual_seed(torch_seeds[i])  # Set Torch Seed
        torch.cuda.manual_seed_all(torch_seeds[i])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # init base model
        model = UNET(
            in_ch=model_params["in_ch"],
            out_ch=model_params["n_classes"],
            bilinear_method=model_params["bilinear_method"],
            momentum=model_params["momentum"],
            enable_dropout=model_params["enable_dropout"],
            dropout_prob=model_params["dropout_prob"],
            enable_pool_dropout=model_params["enable_pool_dropout"],
            pool_dropout_prob=model_params["pool_dropout_prob"],
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        if method == "Laplace":
            # It is neccesary to instaniate the object here, as it requires grad to compute the optimal prior
            LaplaceFitter = Laplace(
                model,
                binary,
                train_loader,
                hessian_method="Exact",
                method="MC",
                n_samples=n_forward_passes,
                prior=torch.tensor(1),
                validate_args=False,
                device=device,
                dataset=dataset,
            )
            LaplaceFitter.prepare_laplace()

        with torch.no_grad():
            model.eval()

            # Normal Predictions
            if method in ["Normal", "BatchNorm"]:
                assert len(models) == 1, "Expected a length of 1 when using method=='Normal'"
                # Iterate Through data_loader
                images_vec = []
                masks_vec = []
                predictions_vec = []
                prediction_idx = []

                for batch in tqdm(data_loader):
                    images, masks, idx = batch
                    images = images.unsqueeze(1) if dataset != "warwick" else images
                    images = images.to(device=device, dtype=torch.float32)
                    masks = masks.type(torch.LongTensor)
                    if model.out_ch > 1:
                        masks = masks.squeeze(1)
                    masks = masks.to(device)
                    predictions = model(images)
                    # predictions = torch.sigmoid(predictions)

                    images_vec.append(images)
                    masks_vec.append(masks)
                    predictions_vec.append(predictions)
                    prediction_idx.append(idx.clone().detach())

                images = torch.vstack(images_vec)
                masks = torch.vstack(masks_vec)
                predictions = torch.vstack(predictions_vec)
                prediction_idx = torch.cat(prediction_idx)

                assert sum(
                    torch.tensor(list(predictions.shape)).eq(
                        torch.tensor([masks.shape[0], model.out_ch, masks.shape[1], masks.shape[2]])
                    )
                ) == len(predictions.shape)
                return (images, masks, predictions, prediction_idx)

            elif method == "MCD":
                assert len(models) == 1, "Expected a length of 1 when using method=='MCD'"
                model.apply(enable_MCDropout)

                images_vec = []
                masks_vec = []
                predictions_vec = []
                prediction_idx = []

                for batch in tqdm(data_loader):
                    images, masks, idx = batch
                    images = images.unsqueeze(1) if dataset != "warwick" else images
                    images = images.to(device=device, dtype=torch.float32)
                    masks = masks.type(torch.LongTensor)
                    if model.out_ch > 1:
                        masks = masks.squeeze(1)
                    masks = masks.to(device)
                    predictions = torch.stack(
                        [model(images).permute(0, 2, 3, 1) for _ in range(n_forward_passes)], dim=1
                    )
                    # predictions = torch.sigmoid(predictions)

                    images_vec.append(images)
                    masks_vec.append(masks)
                    predictions_vec.append(predictions)
                    prediction_idx.append(idx.clone().detach())

                images = torch.vstack(images_vec)
                masks = torch.vstack(masks_vec)
                predictions = torch.vstack(predictions_vec)
                prediction_idx = torch.cat(prediction_idx)
                assert sum(
                    torch.tensor(list(predictions.shape)).eq(
                        torch.tensor(
                            [
                                masks.shape[0],
                                n_forward_passes,
                                masks.shape[1],
                                masks.shape[2],
                                model.out_ch,
                            ]
                        )
                    )
                ) == len(predictions.shape)
                assert ((predictions.sum(dim=(2, 3, 4))).std(dim=1)).mean() != 0

            elif method == "DeepEnsemble":
                images_vec = []
                masks_vec = []
                predictions_vec = []
                prediction_idx = []
                for batch in tqdm(data_loader):
                    images, masks, idx = batch
                    images = images.unsqueeze(1) if dataset != "warwick" else images
                    images = images.to(device=device, dtype=torch.float32)
                    masks = masks.type(torch.LongTensor)
                    if model.out_ch > 1:
                        masks = masks.squeeze(1)
                    masks = masks.to(device)
                    predictions = model(images)
                    predictions = predictions.permute(0, 2, 3, 1)
                    # predictions = torch.sigmoid(predictions)

                    images_vec.append(images)
                    masks_vec.append(masks)
                    predictions_vec.append(predictions)
                    prediction_idx.append(idx.clone().detach())

                images = torch.vstack(images_vec)
                masks = torch.vstack(masks_vec)
                predictions = torch.vstack(predictions_vec)
                prediction_idx = torch.cat(prediction_idx)
                ensemble.append(predictions)

            elif method == "Laplace":
                images, masks, predictions, prediction_idx = LaplaceFitter.get_predictions(
                    data_loader
                )
                assert sum(
                    torch.tensor(list(predictions.shape)).eq(
                        torch.tensor(
                            [
                                masks.shape[0],
                                n_forward_passes,
                                masks.shape[1],
                                masks.shape[2],
                                model.out_ch,
                            ]
                        )
                    )
                ) == len(predictions.shape)
                assert ((predictions.sum(dim=(2, 3, 4))).std(dim=1)).mean() != 0

    if method == "DeepEnsemble":
        ensemble = torch.stack(ensemble, dim=1)
        assert sum(
            torch.tensor(list(ensemble.shape)).eq(
                torch.tensor(
                    [masks.shape[0], len(models), masks.shape[1], masks.shape[2], model.out_ch]
                )
            )
        ) == len(ensemble.shape)
        return (images, masks, ensemble, prediction_idx)
    elif method in ["MCD", "Laplace"]:
        return (images, masks, predictions, prediction_idx)
    else:
        print(
            f"Recieved {method}, which is not valid. Expected either ('Normal','MCD','DeepEnsemble','Laplace')"
        )
        return -1
