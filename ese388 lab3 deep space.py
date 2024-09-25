import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

df = pd.read_csv('/Users/aahyseni/Downloads/DeepSpaceData.csv')

pd.set_option('display.expand_frame_repr', False)

# Print some stuff
print(df.columns.values)
print(df.ndim)
print(df.shape)
print(df.info())
dStats = df.describe()

dfM = df[["X1", "X2"]]
scaler = StandardScaler()
scaled_dfM = scaler.fit_transform(dfM)

epsA = 0.3
MinPts = 5
db = DBSCAN(eps=epsA, min_samples=MinPts).fit(scaled_dfM)

labels = db.labels_
NClusters = len(set(labels)) - (1 if -1 in labels else 0)
dfM["Cluster"] = labels

plt.figure(1)
fa = sns.pairplot(data=dfM, hue="Cluster", palette="Set2")
plt.show()
fa.savefig('DeepSpace_DBSCAN_Clusters.pdf')