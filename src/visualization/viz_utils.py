class styles:
    csfont = {"fontname": "sans-serif"}
    dpi_level = 600


import numpy as np
from src.config import Config
import json


def load_ColorMapper(map_title="ColorMapper"):
    with open(f"data/color_mapping/{map_title}.json", "r") as f:
        ColorMapper = json.load(f)
    return ColorMapper


# ColorMapper()
# {'datasets':color,'binary':color}


# with open('data/color_mapping/test_color.json','r') as f:
#     ColorMapper = json.load(f)


def viz_mask_helper(mask, dataset_type):
    ColorMapper = load_ColorMapper()
    mask = mask.permute(1, 2, 0)  # change to [height, width, n_classes]
    seg_arr = mask.argmax(axis=2)
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    seg_img = np.zeros((output_height, output_width, 3))

    # for c in range(50):
    #     seg_img[:, :, 0] += ((seg_arr[:, :] == c)*(ColorMapper['warwick'][str(c)][0])).numpy()
    #     seg_img[:, :, 1] += ((seg_arr[:, :] == c)*(ColorMapper['warwick'][str(c)][1])).numpy()
    # seg_img[:, :, 2] += ((seg_arr[:, :] == c)*(ColorMapper['warwick'][str(c)][2])).numpy()

    for c in range(Config.n_classes[dataset_type]):
        # for c in range(255):
        seg_img[seg_arr == c] = tuple(ColorMapper[dataset_type][str(c)])

    return seg_img.astype(np.uint8)


# new color palette style
# viridis(6)

# create n_classes colors from viridis
# identify background index
# save colors to json

# create to function here


# def create_color_map():
#     names = list(Config.n_classes.keys())
#     number_classes = list(Config.n_classes.values())

#     ColorMapper = {}

#     for i, name in enumerate(names):
#         colors = viridis(number_classes[i])
#         colors_rgb = [hex_to_rgb(x) for x in colors]
#         mapper = {}
#         for cls in range(number_classes[i]):
#             mapper[cls] = colors_rgb[cls]
#         ColorMapper[name] = mapper
#     with open("data/color_mapping/ColorMapper_new.json", "w") as f:
#         json.dump(ColorMapper, f)
#     print("Succesfully Created Color Mappings")


# create_color_map()


# colors = viridis(255)
# colors_rgb = [hex_to_rgb(x) for x in colors]
# mapper = {}
# for cls in range(255):
#     mapper[cls] = colors_rgb[cls]
# ColorMapper['membrane'] = mapper
# with open('data/color_mapping/test_color.json','w') as f:
#     json.dump(ColorMapper,f)
# print("Succesfully Created Color Mappings")

# ColorMapper = load_ColorMapper()
