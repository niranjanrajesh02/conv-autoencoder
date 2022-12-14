from dataset import load_data
from CAE import CAE
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image
import numpy as np

BATCH_SIZE = 32

RESULTS_PATH = "/home/niranjan.rajesh_ug23/TNBC/Results/ConvAE"

x, y = load_data()

print(x.shape, y.shape)
code_dims = [10, 50, 100, 200, 500, 1000, 2000]

for cd in code_dims:
    model = CAE(input_shape=x.shape[1:], code_dim=cd)
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')
    hist = model.fit(x, x, batch_size=BATCH_SIZE, epochs=30)

    out = model.predict(x)

    counter = 1

    sns.set(rc={'figure.figsize':(12,25)})
    fig =  plt.figure()
    rows = 4
    for i in range(rows):
        ind = random.randint(0, 1200)
        X_img = Image.fromarray(np.uint8(x[ind] *255)).convert('RGB')
        R_img = Image.fromarray(np.uint8(out[ind] *255)).convert('RGB')
        ax1 = fig.add_subplot(rows,2,counter)
        ax1.title.set_text("Input Image")
        ax1 = ax1.imshow(X_img)
        plt.grid(False)
        counter += 1
        ax2 = fig.add_subplot(rows,2,counter)
        ax2.title.set_text("Reconstructed Image")
        ax2 = ax2.imshow(R_img)
        plt.grid(False)
        counter += 1
    fig.suptitle(f"Image Reconstruction with Code Dimension {cd}", fontsize=20)
    fig.savefig(f'{RESULTS_PATH}/Reconstructed_Patches_cd{cd}.png')
    print(f"Code Dimension {cd} done!")
print("Reconstruction Figs saved successfully!")