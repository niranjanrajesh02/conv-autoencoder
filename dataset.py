import os
import numpy as np
from PIL import Image

global DATA_PATH  
# DATA_PATH = "/storage/tnbc/gen1_label/224_gen1"
DATA_PATH = "C:/Niranjan/Ashoka/Research/TNBC/Data/224_gen1"

global claass_names
class_names = os.listdir(DATA_PATH)


def load_data():
    x = []
    y = []
    for n in range(len(class_names)):
        class_path = f'{DATA_PATH}/{class_names[n]}'
        patches_in_class = os.listdir(class_path)
        for p in range(len(patches_in_class)):
            img_path = f'{class_path}/{patches_in_class[p]}'
            x.append(np.asarray(Image.open(img_path)))
            y.append(n)
    x = np.array(x)
    x = x/255.
    y = np.array(y)
    return x, y