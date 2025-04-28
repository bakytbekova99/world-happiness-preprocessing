import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('world_happiness_normalized.csv')
features = df[['Economy', 'Social support', 'Health']]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(features_scaled, method='ward'))
plt.title('Dendrogram for World Happiness Data')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distance')
plt.show()

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(features_scaled)
df['Cluster'] = y_hc
df.to_csv('world_happiness_with_clusters.csv', index=False)

print(df.head())
