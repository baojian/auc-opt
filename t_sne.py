import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import TSNE
import time
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.decomposition import PCA

from data_preprocess import data_preprocess_australian
from data_preprocess import data_preprocess_diabetes
from data_preprocess import data_preprocess_breast_cancer
from data_preprocess import data_preprocess_splice


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()
    return f, ax, sc, txts


def main():
    x_train, y_train = fetch_openml('mnist_784', version=1, return_X_y=True)
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    RS = 123

    # Subset first 20k data points to visualize
    x_subset = x_train[0:20000]
    y_subset = y_train[0:20000]

    print(np.unique(y_subset))
    time_start = time.time()
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(x_subset)

    print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))
    pca_df = pd.DataFrame(columns=['pca1', 'pca2', 'pca3', 'pca4'])

    pca_df['pca1'] = pca_result[:, 0]
    pca_df['pca2'] = pca_result[:, 1]
    pca_df['pca3'] = pca_result[:, 2]
    pca_df['pca4'] = pca_result[:, 3]
    print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

    top_two_comp = pca_df[['pca1', 'pca2']]  # taking first and second principal component

    fashion_scatter(top_two_comp.values, y_subset)  # Visualizing the PCA output

    time_start = time.time()

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(x_subset)

    fashion_tsne = TSNE(random_state=RS).fit_transform(x_subset)
    fashion_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    fashion_scatter(fashion_pca_tsne, y_subset)


if __name__ == '__main__':
    pass
