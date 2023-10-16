from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['Label'] = y
df['Species'] = df['Label'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

pca = PCA()
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
df = pd.merge(df, pca_df, right_index=True, left_index=True)

print('Explained Variance Ratio')
for i in range(4):
    print('PC{}: {}'.format(i + 1, pca.explained_variance_ratio_[i]))

sns.stripplot(x="PC1", y="Species", data=df, jitter=True)
plt.title('Iris Data Visualized in One Dimension')
plt.show()

precent_of_variance_explained = 0.95
pca = PCA(n_components=precent_of_variance_explained)
pca_data = pca.fit_transform(X)
print("{} Principal Components are required to explain {} of the variation in this data.".format(pca.n_components_, precent_of_variance_explained))

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
dim = np.arange(len(cumsum)) + 1
plt.plot(dim, cumsum, '-', lw=3)
plt.xlabel('Dimensions')
plt.ylabel('Variance Explained')
plt.title('Selecting the right number of dimensions')
plt.xticks([1, 2, 3, 4])
plt.ylim([0, 1.1])
plt.show()

sns.lmplot(x='PC1', y='PC2', data=df, hue='Species', fit_reg=False)
plt.title('Iris Data Visualized in Two Dimensions')
plt.show()
