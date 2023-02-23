import os
import numpy as np
from PIL import Image
from helper import DATA_PATH


global claass_names
class_names = os.listdir(DATA_PATH)


def load_data():
    x = []
    y = []

    target_map = {}
    counter = 0
    for c in class_names:
        target_map[c] = counter
        counter += 1

    for n in range(len(class_names)):
        class_path = f'{DATA_PATH}/{class_names[n]}'
        patches_in_class = os.listdir(class_path)
        for p in range(len(patches_in_class)):
            img_path = f'{class_path}/{patches_in_class[p]}'
            x.append(np.asarray(Image.open(img_path)))
            y.append(target_map[n])
    x = np.array(x)
    x = x/255.
    y = np.array(y)
    return x, y
