import os
import psycopg2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dbclient import DBClient
from vectorclient import get_embeddings, reflect_vector
from clusterclient import get_clusters_kmeans, get_optimal_clusters_kmeans_elbow
from gptclient import getCommonTheme, generate_cluster_names_many

connectionString = os.environ['pgVectorConnectionString']

def chooseK(size: int):
    kMap = {
        5: 2,
        10: 3,
        50: 5,
        200: 15,
        2000: 25
    }

    keys = list(kMap.keys())
    keys.sort()
    k = 35
    for key in keys:
        if size <= key:
            k = kMap[key]
            break
    return k

def getThemes(text: list, k: str = None, commonConcept: str = None):

    with psycopg2.connect(connectionString) as conn:
        with DBClient(conn) as dbClient:

            if k is None:
                k = chooseK(len(text))
                      
            # Vector processing
            vectors = get_embeddings(text, dbClient)
            if(commonConcept is not None):
                commonVector = get_embeddings([commonConcept], dbClient)[0]
                vectors = np.apply_along_axis(reflect_vector, 1, vectors, commonVector)

            if k == "auto":
                labels, descriptions, centroids = get_optimal_clusters_kmeans_elbow(dbClient, vectors)
            else:
                k = int(k)
                labels, descriptions, centroids = get_clusters_kmeans(dbClient, vectors, k)

            similarity = cosine_similarity(vectors, centroids)

            # GPT labelling
            tasks = []
            for(i, label) in enumerate(descriptions):
                # get indicies of labels where value is i
                clusterIndices = np.where(labels == i)
                clusterSimilarity = similarity[clusterIndices]
                clusterText = np.array(text)[clusterIndices]
                similarityScores = clusterSimilarity[:, i]
                sortedIndicies = np.argsort(similarityScores)[::-1]
                top3 = sortedIndicies[:3]
                remaining = sortedIndicies[3:]
                sample_count = min(2, len(remaining))
                np.random.seed(42)
                samples = np.concatenate((clusterText[top3], np.random.choice(clusterText[remaining], sample_count, replace=False)))
                task = {"labels": label, "samples": samples, "additionalInstructions": ""}
                tasks.append(task)

            commonTheme = getCommonTheme(descriptions)
            clusterNames = generate_cluster_names_many(tasks)

            # Report generation
            themeReportItems = []    
            for(i, desc) in enumerate(descriptions):
                # get indicies of labels where value is i
                clusterIndices = np.where(labels == i)
                clusterSimilarity = similarity[clusterIndices]
                clusterText = np.array(text)[clusterIndices]
                similarityScores = clusterSimilarity[:, i]
                top5 = np.argsort(similarityScores)[::-1][:5]
                report = {
                    "name": clusterNames[i],
                    "keywords": desc,
                    "topSamples": list(clusterText[top5]),
                    "embeddingModel" : "text-embedding-ada-002",
                    "embeddingVector": list(centroids[i]),
                }
                themeReportItems.append(report)

            return {
                "commonTheme": commonTheme,
                "themes": themeReportItems
            }