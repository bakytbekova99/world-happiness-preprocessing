
# World Happiness Report Preprocessing Code

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Load data (assume already combined across 2020–2024)
# For actual run, load the combined dataset from multiple CSVs
# Sample pre-loaded dataframe: df_cleaned

# 1. Data Cleaning - drop nulls and rename columns (already done)

# 2. Data Normalization
features = ['happiness_score', 'economy_(gdp_per_capita)', 'social_support',
            'healthy_life_expectancy', 'freedom_to_make_life_choices',
            'generosity', 'perceptions_of_corruption']
scaler = MinMaxScaler()
normalized = scaler.fit_transform(df_cleaned[features])
df_normalized = pd.DataFrame(normalized, columns=features)

# 3. PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_normalized)
df_cleaned['PCA1'] = pca_result[:, 0]
df_cleaned['PCA2'] = pca_result[:, 1]

# 4. Linear Regression
X = df_cleaned[['economy_(gdp_per_capita)', 'social_support', 'healthy_life_expectancy']]
y = df_cleaned['happiness_score']
reg = LinearRegression().fit(X, y)

# 5. Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(df_normalized)

# 6. Data Discretization
df_cleaned['happiness_level'] = pd.cut(
    df_cleaned['happiness_score'],
    bins=[0, 4, 6, 8, 10],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# 7. Histogram
sns.histplot(df_cleaned['happiness_score'], bins=20, kde=True)
plt.title('Histogram of Happiness Scores')
plt.xlabel('Happiness Score')
plt.ylabel('Frequency')
plt.savefig('happiness_score_histogram.png')
plt.close()
