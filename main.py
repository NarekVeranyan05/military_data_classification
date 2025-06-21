import numpy as np
import pandas as pd
import imageio.v3 as iio
from pathlib import Path

def readData() -> pd.DataFrame:
    dirs = ["images/train/bmp", "images/train/btr", "images/train/cars", "images/train/grad", "images/train/howitzer", "images/train/tank"]
    img_paths = [Path(d) for d in dirs]
    paths = []
    for path in img_paths:
        paths += list(path.glob('*.jpeg'))
        paths += list(path.glob('*.jpg'))
        paths += list(path.glob('*.png'))

    images = pd.DataFrame([iio.imread(path).flatten() for path in paths])
    return images

def main():
    images = readData()


main()