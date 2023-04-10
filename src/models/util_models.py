from omegaconf import OmegaConf
import numpy as np


class EarlyStopping:
    def __init__(self, tolerance=20, best_val_loss=np.inf):
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


def init_params(args, cfg):
    del args["cfg"]

    config_string = "\n--------------------------\n"
    config_string += "Parameter Configuration: \n"
    config_string += OmegaConf.to_yaml(cfg)
    for key in args:
        config_string += f"{key}: {args[key]}\n"
    config_string += "--------------------------"
    return config_string
