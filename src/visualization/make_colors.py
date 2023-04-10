from src.config import Config
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis, viridis
import json
import argparse
import os

try:
    map_title = os.environ["map_title"]
except:
    map_title = "SegmentColor"


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip("#")
    return list(int(hex_code[x : x + 2], 16) for x in (0, 2, 4))


def create_color_map(map_title):
    print(map_title)
    names = list(Config.n_classes.keys())
    number_classes = list(Config.n_classes.values())
    # Add a default binary color map
    names.append("binary")
    number_classes.append(2)
    ColorMapper = {}

    for i, name in enumerate(names):
        colors = viridis(number_classes[i])
        colors_rgb = [hex_to_rgb(x) for x in colors]
        mapper = {}
        for cls in range(number_classes[i]):
            mapper[cls] = colors_rgb[cls]
        ColorMapper[name] = mapper
    with open(f"data/color_mapping/{map_title}.json", "w") as f:
        json.dump(ColorMapper, f)
    print("Succesfully Created Color Mappings")


if __name__ == "__main__":
    create_color_map(map_title=map_title)
