import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


ionosphere_path = '/Users/aahyseni/Downloads/ionosphere.csv'
ionosphere_data = pd.read_csv(ionosphere_path, header=None)


features = ionosphere_data.iloc[:, :-1]  
target = ionosphere_data.iloc[:, -1]    


scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)


pca = PCA()
pca_data = pca.fit_transform(standardized_features)

explained_variance = pca.explained_variance_ratio_


plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='salmon')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.show()


cumulative_variance = np.cumsum(explained_variance)
s = np.argmax(cumulative_variance >= 0.90) + 1


print(f'Number of components to retain for 90% variance: {s}')
