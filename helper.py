import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from main import RESULTS_PATH


def visualise_patches(in_img, out_img, dim):
    counter = 1
    sns.set(rc={'figure.figsize': (12, 25)})
    fig = plt.figure()
    rows = 4
    patches = [132, 532, 234, 957]
    for i in range(rows):
        ind = patches[i]
        X_img = Image.fromarray(np.uint8(in_img[ind] * 255)).convert('RGB')
        R_img = Image.fromarray(np.uint8(out_img[ind] * 255)).convert('RGB')
        ax1 = fig.add_subplot(rows, 2, counter)
        ax1.title.set_text("Input Image")
        ax1 = ax1.imshow(X_img)
        plt.grid(False)
        counter += 1
        ax2 = fig.add_subplot(rows, 2, counter)
        ax2.title.set_text("Reconstructed Image")
        ax2 = ax2.imshow(R_img)
        plt.grid(False)
        counter += 1

    fig.suptitle(
        f"Image Reconstruction with Code Dimension {dim}", fontsize=20)
    fig.savefig(f'{RESULTS_PATH}/Reconstructed_Patches_cd{dim}.png')
