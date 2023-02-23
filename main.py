from dataset import load_data
from CAE import CAE
from keras.models import Model
from helper import visualise_patches


import matplotlib.pyplot as plt
import random
import numpy as np

BATCH_SIZE = 200


x, y = load_data()

print(x.shape, y.shape)
# code_dims = [50, 100, 200, 500, 1000]
code_dims = [1000]


for cd in code_dims:
    model = CAE(input_shape=x.shape[1:], code_dim=cd)

    model.compile(optimizer='adam', loss='mse')
    hist = model.fit(x, x, batch_size=BATCH_SIZE, epochs=200)

    out = model.predict(x)
    feature_model = Model(inputs=model.input,
                          outputs=model.get_layer(name='embedding').output)

    # visualisng patches
    visualise_patches(x, out, cd)
    print(f"Code Dimension {cd} done!")


print("Reconstruction Figs saved successfully!")
