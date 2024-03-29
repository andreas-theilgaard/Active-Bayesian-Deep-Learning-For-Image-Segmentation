class Config:
    datasets = ["warwick", "DIC_C2DH_Hela", "PhC-C2DH-U373", "membrane"]
    # n_classes = {"warwick": 50, "DIC_C2DH_Hela": 20, "PhC-C2DH-U373": 14, "membrane": 2}
    n_classes = {"warwick": 33, "DIC_C2DH_Hela": 15, "PhC-C2DH-U373": 8, "membrane": 2}

    title_mapper = {
        "DIC_C2DH_Hela": "DIC-Hela",
        "PhC-C2DH-U373": "PhC-U373",
        "membrane": "Membrane",
        "warwick": "GlaS",
    }


import torch


def find_best_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
