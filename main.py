from dataset import load_data
from CAE import CAE
from keras.models import Model
from helper import visualise_patches
from clustering import cluster_KM

import matplotlib.pyplot as plt
import random
import numpy as np

BATCH_SIZE = 200


x, y = load_data()

print(x.shape, y.shape)
code_dims = [50, 100, 200, 500, 1000, 2000]
# code_dims = [1000]
feat_dict = {}

for cd in code_dims:
    model = CAE(input_shape=x.shape[1:], code_dim=cd)

    model.compile(optimizer='adam', loss='mse')
    hist = model.fit(x, x, batch_size=BATCH_SIZE, epochs=200)

    out = model.predict(x)
    feature_model = Model(inputs=model.input,
                          outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    print('feature shape=', features.shape)
    feat_dict[cd] = features
    # visualisng patches
    # visualise_patches(x, out, cd)

    print(f"Code Dimension {cd}  done!")


for c in feat_dict:

    features = feat_dict
    # cluster features
    print(f'Clustering Results for Dim {c}:')
    cluster_KM(features, y)
