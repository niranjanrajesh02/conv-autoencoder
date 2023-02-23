from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np


def cluster_KM(features, labels):
    print("Clustering ...")
    km = KMeans(n_clusters=9)
    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)
    print(f"NMI: {normalized_mutual_info_score(pred,labels)}")
    print("Clustering Complete!")
