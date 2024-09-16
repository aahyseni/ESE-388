#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:34:58 2024

@author: aahyseni
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans



file_path = '/Users/aahyseni/Downloads/Raisin_Dataset.csv'  
raisin = pd.read_csv(file_path)
sns.pairplot(data=raisin, hue='Class')
plt.title('scatter plot')
plt.show()




file_path_2 = '/Users/aahyseni/Downloads/DeepSpaceData.csv'  
DeepSpace = pd.read_csv(file_path_2)
sns.pairplot(data=DeepSpace)
plt.title('Deep Space Sctter Plot')
plt.show()





raisinX = raisin[["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"]]
raisinY = raisin[["Class"]]

km = KMeans(n_clusters=3)
km.fit(raisinX)

centers = km.cluster_centers_
print(centers)

plt.scatter(raisinX["Area"], raisinX["Perimeter"], c=km.labels_)
plt.title('Cluster Raisin')
plt.show()

km2 = KMeans(n_clusters=3)
km2.fit(DeepSpace)
plt.scatter(DeepSpace['X1'], DeepSpace['X2'], c=km2.labels_)
plt.show()