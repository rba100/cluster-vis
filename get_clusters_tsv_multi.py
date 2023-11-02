import psycopg2
import pandas as pd
import openai
import numpy as np
from sklearn.cluster import KMeans
from adaclient import get_embeddings
from vectordbclient import get_closest_words;
from gptclient import name_clusters_array, name_clusters_summary
from scipy.stats import chi2_contingency

# Connect to the database
conn = psycopg2.connect(host='localhost', database='postgres', user='postgres', password='postgres')
cursor = conn.cursor()

# Load data from input.csv (except its tab separated)
#data = pd.read_csv("input.csv")

data_col_name = 'text'
data_cat_col_names = ['Gender','Region','Satisfaction Rating (Categorical)','Occupation', 'Experience with the brand (Text)','Last Purchase Delivery channel (for brand)']

data = pd.read_csv("input.csv", sep='\t')
data = data[data[data_col_name].apply(lambda x: isinstance(x, str) and x != "")]

embeddings_result = get_embeddings(data[data_col_name].tolist(), conn)
data['embedding'] = embeddings_result.tolist()
embeddings_array = np.array(data['embedding'].tolist())

# Perform clustering
n_clusters = 10
#kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init="auto", random_state=42)
data['cluster'] = kmeans.fit_predict(embeddings_array)
cluster_centers = kmeans.cluster_centers_

# Normalize the cluster centers
cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

stringbuilder = ""
closest_words_list = []
closest_words_list_str = []
cluster_names = {}

# Lookup nearest 5 words for each normalized cluster center
for i, center in enumerate(cluster_centers):
    closest_words = get_closest_words(center, cursor)
    closest_words_list.append(closest_words)
    msg = f"Cluster {i+1} closest words: {', '.join(closest_words)}"
    closest_words_list_str.append(', '.join(closest_words))
    print(msg)
    stringbuilder = stringbuilder + f"Cluster {i+1} closest words: {', '.join(closest_words)}\n"

# Use GPT to name Clusters
cluster_names = name_clusters_array(closest_words_list_str)

summary = name_clusters_summary(stringbuilder)
print(summary)

# Detect correlations between clusters and demographics

for column in data_cat_col_names:
    contingency = pd.crosstab(data['cluster'], data[column].astype('category'))
    chi2, p, _, expected = chi2_contingency(contingency)
    
    print(f"\nAssociation between clusters and {column}:")
    print(f"P-value: {p:.10f}")
    if p < 0.05:  # 5% level of significance
        for i, (actual_counts, expected_counts) in enumerate(zip(contingency.values, expected)):
            diff = actual_counts - expected_counts
            
            # Just as an example, if the actual count is higher than expected by more than 20%, consider it significant
            significant_indices = np.where(diff > 0.2 * expected_counts)[0]
            
            if significant_indices.size > 0:
                associated_categories = data[column].astype('category').cat.categories[significant_indices].tolist()
                #print(f"  - Cluster {i+1} ({', '.join(closest_words_list[i])}) has higher counts for {column} categories: {', '.join(associated_categories)}")
                print(f"  - Cluster {i+1} ({cluster_names[i]}) has higher counts for {column} categories: {', '.join(associated_categories)}")
    else:
        print(f"No significant association detected for {column}.")


# Close the connection
conn.close()
