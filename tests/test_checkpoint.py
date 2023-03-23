import torch
from src.models.model import UNET, init_weights
from src.config import find_best_device
from src.data.dataloader import train_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def simulated_train_loop():
    dataset = "membrane"
    device = find_best_device()
    torch.manual_seed(17)
    model = UNET(
        in_ch=1,
        out_ch=1,
        bilinear_method=False,
        momentum=0.9,
        enable_dropout=False,
        dropout_prob=0.5,
        enable_pool_dropout=False,
        pool_dropout_prob=0.5,
    )
    model.apply(init_weights)
    model.to(device)
    (
        train_loader,
        val_loader,
        unlabeled_loader,
        train_idx,
        val_idx,
        unlabeled_pool_idx,
    ) = train_split(
        train_size=0.01,
        dataset=dataset,
        batch_size=4,
        to_binary=True,
        num_workers=0,
        seed=41,
    )
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.1, threshold=0.001
    )

    # Now train loop
    for epoch in range(1):
        train_loop = tqdm(train_loader, leave=False)  # Progress bar for the training data
        for batch_number, batch in enumerate(train_loop):
            images, masks, idx = batch
            images = images.unsqueeze(1) if dataset != "warwick" else images
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.type(torch.LongTensor)
            if model.out_ch > 1:
                masks = masks.squeeze(1)
            masks = masks.to(device)
            # get predictions
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss

            loss.backward()
            optimizer.step()
    model.eval()
    eval_preds = model(images)
    print(eval_preds.shape)
    saved_eval_preds = (eval_preds[0][0][0][0], eval_preds[0][0][34][17], eval_preds[0][0][56][44])
    torch.save(model.state_dict(), f"models/checkpoint_test.pth")
    torch.save(images, "results/checkpoint_images.pth")
    return saved_eval_preds


def simulated_eval_loop():
    device = find_best_device()
    print(f"Running on Device {device}")
    train_out = simulated_train_loop()

    model = UNET(
        in_ch=1,
        out_ch=1,
        bilinear_method=False,
        momentum=0.9,
        enable_dropout=False,
        dropout_prob=0.5,
        enable_pool_dropout=False,
        pool_dropout_prob=0.5,
    )
    model.load_state_dict(torch.load("models/checkpoint_test.pth"))
    model.eval()
    model.to(device)
    img_test = torch.load("results/checkpoint_images.pth")
    eval_preds = model(img_test)
    preds = (eval_preds[0][0][0][0], eval_preds[0][0][34][17], eval_preds[0][0][56][44])
    print(preds[0], train_out[0])
    print(preds[1], train_out[1])
    print(preds[2], train_out[2])
    assert preds[0] == train_out[0]
    assert preds[1] == train_out[1]
    assert preds[2] == train_out[2]


if __name__ == "__main__":
    simulated_eval_loop()
