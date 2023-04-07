import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, tolerance=15, best_val_loss=np.inf):
        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.best_val_loss = best_val_loss

    def __call__(self, validation_loss):
        if validation_loss < self.best_val_loss:
            self.counter = 0
            self.best_val_loss = validation_loss
        else:
            self.counter += 1
        if self.counter >= self.tolerance:
            self.early_stop = True


def toy_data(n_samples=500):
    np.random.seed(0)
    X = np.random.normal(size=(n_samples, 1)).reshape(-1, 1)
    y = np.random.normal(np.cos(5 * X) / (np.abs(X) + 1), 0.1).ravel()
    return (X, y)


class DatasetClass(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def get_toy_data(n_samples):
    X, y = toy_data(n_samples=n_samples)
    train = DatasetClass(X, y)
    train_data = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True, num_workers=0)
    return train_data


class NN(nn.Module):
    def __init__(self, enable_dropout=False, dropout_prob=0.5):
        super().__init__()

        layers = [nn.Linear(1, 1024)]
        if enable_dropout:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(1024, 512))
        if enable_dropout:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(512, 128))
        if enable_dropout:
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(128, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def train(enable_dropout_var=False, n_samples=500, epochs=100, device="cpu", seed=7):
    train_data = get_toy_data(n_samples=n_samples)
    model = NN(enable_dropout=enable_dropout_var)
    model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    EarlyStopper = EarlyStopping(tolerance=15)
    torch.manual_seed(seed)

    train_loss_global = []
    for _ in tqdm(range(epochs)):
        train_loss_local = []
        for batch in train_data:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred.squeeze(1), y)
            loss.backward()
            optimizer.step()
            train_loss_local.append(loss.item())
        local_mean_loss = torch.tensor(train_loss_local).mean().item()
        train_loss_global.append(local_mean_loss)
        EarlyStopper(local_mean_loss)
        # print(train_loss_global)
        if EarlyStopper.early_stop:
            print("Now we stop")
            break

    torch.save(model.state_dict(), f"playground/train_reg_model_{seed}.pt")
    model.eval()
    print(f"Predictions in train loop {model(torch.tensor([[0.3],[0.7896]]))}")


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def deep_ensemble_predict(seeds, n_samples=500, device="cpu"):
    preds_array = np.zeros((n_samples, len(seeds)))
    X, y = toy_data(n_samples=n_samples)
    X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    for i in range(len(seeds)):
        model = NN()
        model.load_state_dict(torch.load(f"playground/train_reg_model_{seeds[i]}.pt"))
        model.to(device)
        model.eval()
        print(f"Predictions in deep_ensemvle loop {model(torch.tensor([[0.3],[0.7896]]))}")
        with torch.no_grad():
            predictions = model(X)
            predictions = predictions.detach().cpu().numpy()
            predictions = predictions.ravel()
            preds_array[:, i] = predictions

    print(preds_array)
    print(preds_array[:, i].shape)
    print(f"RMSE: {mean_squared_error(y,np.mean(preds_array,axis=1))}")
    plt.plot(X[:, 0], y, "r.")
    for i in range(len(seeds)):
        plt.plot(X[:, 0], preds_array[:, i], "b.", alpha=1 / 20)
    plt.show()


def MC_predict(
    mc_samples=50, n_samples=500, model_path="playground/train_reg_model_10.pt", device="cpu"
):
    torch.manual_seed(10)
    model = NN(enable_dropout=True)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    X, y = toy_data(n_samples=n_samples)
    X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    preds_array = np.zeros((n_samples, mc_samples))
    # for i in range(mc_samples):
    with torch.no_grad():
        model.eval()
        print(f"Predictions in MC_predict loop {model(torch.tensor([[0.3],[0.7896]]))}")
        model.apply(enable_dropout)
        for i in range(mc_samples):
            predictions = model(X)
            predictions = predictions.detach().cpu().numpy()
            predictions = predictions.ravel()
            preds_array[:, i] = predictions
    print(preds_array)
    print(preds_array[:, i].shape)
    print(f"RMSE: {mean_squared_error(y,np.mean(preds_array,axis=1))}")
    plt.plot(X[:, 0], y, "r.")
    for i in range(mc_samples):
        plt.plot(X[:, 0], preds_array[:, i], "b.", alpha=1 / 200)
    plt.show()


type_ = "DE"
# type_ = "MCD"

if __name__ == "__main__":
    if type_ == "DE":
        # Deep Ensemble
        seeds = [7, 2, 3]
        for seed in seeds:
            torch.manual_seed(seed)
            train(enable_dropout_var=False, n_samples=500, epochs=100, device="cpu", seed=seed)
        print("Deep Ensemble Predictions")
        deep_ensemble_predict(seeds, n_samples=500, device="cpu")
    elif type_ == "MCD":
        # MC Dropout
        torch.manual_seed(10)
        train(enable_dropout_var=True, n_samples=500, epochs=100, device="cpu", seed=10)
        print("MC Predictions")
        MC_predict(mc_samples=50, n_samples=500)
