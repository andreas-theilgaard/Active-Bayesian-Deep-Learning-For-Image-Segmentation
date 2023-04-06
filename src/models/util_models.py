from omegaconf import OmegaConf


def init_params(args, cfg):
    del args["cfg"]

    config_string = "\n--------------------------\n"
    config_string += "Parameter Configuration: \n"
    config_string += OmegaConf.to_yaml(cfg)
    for key in args:
        config_string += f"{key}: {args[key]}\n"
    config_string += "--------------------------"
    return config_string
