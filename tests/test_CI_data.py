import pandas as pd

def test_CI_data():
    data = pd.read_json('data/color_mapping/test_color.json')
    assert data.shape == (255,1)