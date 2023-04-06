# from src.models.model_with_linear import UNET,init_weights
from src.models.model_lin import UNET, init_weights

# from src.models.model import UNET,init_weights
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.models.awesome_model_utils import CollectMetrics
from src.data.dataloader import train_split, data_from_index
from torch.nn import BCEWithLogitsLoss
from backpack.extensions import (
    GGNMP,
    HMP,
    KFAC,
    KFLR,
    KFRA,
    PCHMP,
    BatchDiagGGNExact,
    BatchDiagGGNMC,
    BatchDiagHessian,
    BatchGrad,
    BatchL2Grad,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    SqrtGGNExact,
    SqrtGGNMC,
    SumGradSquared,
    Variance,
)


def train(
    batch_size=4,
    learning_rate=0.001,
    epochs=100,
    momentum=0.9,
    beta0=0.5,
    train_size=0.99,
    dataset="PhC-C2DH-U373",
    device="cpu",
    validation_size=0.33,
    binary=True,
    enable_dropout=False,
    dropout_prob=0.5,
    enable_pool_dropout=True,
    pool_dropout_prob=0.5,
    bilinear_method=False,
    model_method="PoolDropout",
    seed=261,
    torch_seed=17,
    linear_last_layer=True,
    conv_weight_init=False,
):

    torch.manual_seed(torch_seed)

    # Load training and validation data
    (
        train_loader,
        val_loader,
        unlabeled_loader,
        train_idx,
        val_idx,
        unlabeled_pool_idx,
    ) = train_split(
        train_size=train_size,
        dataset=dataset,
        batch_size=batch_size,
        to_binary=binary,
        num_workers=0,
        seed=seed,
    )

    n_classes = 1 if binary else Config.n_classes[dataset]  # out channels to use for U-Net
    DEVICE = device  # Device to use
    in_ch = 1 if dataset != "warwick" else 3
    print(linear_last_layer)
    print(enable_pool_dropout)
    # Get model and intilize weights
    model = UNET(
        1,
        1,
        bilinear_method=bilinear_method,
        momentum=momentum,
        enable_dropout=enable_dropout,
        dropout_prob=0.5,
        enable_pool_dropout=enable_pool_dropout,
        pool_dropout_prob=0.5,
        linear_last_layer=True,
        conv_weight_init=conv_weight_init,
    )
    # model = UNET(1,1,bilinear_method=bilinear_method,momentum=momentum,enable_dropout=False,dropout_prob=0.5,enable_pool_dropout=False,pool_dropout_prob=0.5)
    model.apply(init_weights)
    for name, param in model.named_parameters():
        if name == "out.weight":
            weight_param = param
        # elif name == 'out.bias':
        #    bias_param = param
    # print(bias_param)
    print(weight_param)
    torch.save(weight_param, "weights/weights_out_seed261.pth")
    # torch.save(bias_param,'weights/bias_out_seed261.pth')
    model.to(device)

    # Intilize criterion and optimizer
    loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta0, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.1, threshold=0.001
    )

    MetricsTrain = CollectMetrics(device=device, out_ch=model.out_ch)
    MetricsValidate = CollectMetrics(device=device, out_ch=model.out_ch)
    # Now train loop
    for epoch in range(epochs):
        train_loop = tqdm(train_loader)  # Progress bar for the training data
        for batch_number, batch in enumerate(train_loop):
            images, masks, idx = batch
            images = images.unsqueeze(1) if dataset != "warwick" else images
            print(images.shape)
            images = images.to(device=DEVICE, dtype=torch.float32)
            masks = masks.type(torch.LongTensor)
            if model.out_ch > 1:
                masks = masks.squeeze(1)
            masks = masks.to(DEVICE)
            # get predictions
            optimizer.zero_grad()
            predictions = model(images)
            predictions = predictions.reshape(predictions.shape[0], 1, 64, 64)
            loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss

            # Collect metrics
            out_loss, out_Dice = MetricsTrain.GetMetrics(predictions, masks, loss)
            loss.backward()
            optimizer.step()
            train_loop.set_postfix({"loss": out_loss, "train_dice": out_Dice})
            MetricsTrain.AppendToGlobal()

        if epoch > -1:  # epoch%10==0:
            val_loop = tqdm(val_loader)
            with torch.no_grad():
                model.eval()
                for batch_number, batch in enumerate(val_loop):
                    images, masks, idx = batch
                    images = images.unsqueeze(1) if dataset != "warwick" else images
                    images = images.to(device=DEVICE, dtype=torch.float32)
                    masks = masks.type(torch.LongTensor)
                    if model.out_ch > 1:
                        masks = masks.squeeze(1)
                    masks = masks.to(DEVICE)
                    # get predictions
                    predictions = model(images)
                    loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss

                    # Collect metrics
                    out_loss, out_Dice = MetricsValidate.GetMetrics(predictions, masks, loss)
                    val_loop.set_postfix({"loss": out_loss, "val_dice": out_Dice})
                    MetricsValidate.AppendToGlobal()

        model.train()
    torch.save(model.state_dict(), f"models/LaplaceTest.pth")

    return (train_loader, val_loader)  # , unlabeled_loader, train_idx, val_idx, unlabeled_pool_idx)


from backpack import extend, backpack, extensions
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

if __name__ == "__main__":
    train_first = False
    if train_first:
        train_loader, val_loader = train(
            batch_size=4,
            learning_rate=0.001,
            epochs=20,
            momentum=0.9,
            beta0=0.5,
            train_size=0.99,
            dataset="PhC-C2DH-U373",
            device="cpu",
            validation_size=0.33,
            binary=True,
            enable_dropout=False,
            dropout_prob=0.5,
            enable_pool_dropout=False,
            pool_dropout_prob=0.5,
            bilinear_method=False,
            model_method="PoolDropout",
            seed=261,
            torch_seed=17,
            linear_last_layer=True,
            conv_weight_init=False,
        )
    else:
        (
            train_loader,
            val_loader,
            unlabeled_loader,
            train_idx,
            val_idx,
            unlabeled_pool_idx,
        ) = train_split(
            train_size=0.99,
            dataset="PhC-C2DH-U373",
            batch_size=4,
            to_binary=True,
            num_workers=0,
            seed=261,
        )

    images, masks = [], []
    for batch in train_loader:
        images.append(batch[0].unsqueeze(1))
        masks.append(batch[1])

    images = torch.vstack(images)
    masks = torch.vstack(masks)
    print(images.shape)
    print(masks.shape)

    model = UNET(
        in_ch=1,
        out_ch=1,
        bilinear_method=False,
        momentum=0.9,
        enable_dropout=False,
        dropout_prob=0.5,
        enable_pool_dropout=False,
        pool_dropout_prob=0.5,
        linear_last_layer=True,
        conv_weight_init=False,
    )
    model.load_state_dict(torch.load("models/LaplaceTest.pth"))
    model.eval()

    # model =
    extend(model)
    loss_func = extend(BCEWithLogitsLoss(reduction="sum"))
    # loss_func = F.binary_cross_entropy_with_logits()
    print(images.dtype)
    print(masks.dtype)
    preds = model(images).squeeze(1)
    print(preds.shape, preds.dtype)
    loss = loss_func(preds, masks.float())

    print(loss)

    with backpack(extensions.KFAC()):
        loss.backward()
    # with backpack(extensions.KFAC()):
    #   loss.backward()
