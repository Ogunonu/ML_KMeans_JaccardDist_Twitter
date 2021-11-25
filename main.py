import numpy as np
import random
import pandas as pd
from statistics import mean
import collections


class KMeans:
    # initialize datafile
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile, sep='\n', names=['message'])

    # Pre-process the data to be usable
    def preprocess(self):
        self.processed_data = self.raw_input

        # remove tweet id and timestamp
        self.df = self.processed_data['message'].str.split('|', 2).str[2]

        # remove any word that starts with @ symbol
        self.df = self.df.str.replace('(@).([^\s]+) ', '', regex=True)

        # remove any hashtag symbols
        self.df = self.df.str.replace('#', '')

        # remove any URL
        self.df = self.df.str.replace('(http).([^\s]+)', '', regex=True)

        # Convert every word to lowercase
        self.df = self.df.str.lower()

        # setup dataframe for pandas
        self.df = pd.DataFrame(self.df)

        return 0

    # get Jaccard Distance 1 - ((a & b)/(a | b))
    def jaccard_dist(self, a, b):
        a = a.to_string()
        b = b.to_string()
        a = set(a.split())
        b = set(b.split())
        output = float(1 - float(len(a & b)) / len(a | b))
        return output

    # Find Closest cluster
    def find_closest(self, a, b):
        closest_dist = []
        closest = []
        # loops for each data point in array
        for i in range(len(self.df)):
            dist = []
            # loops for all centroids
            for j in range(len(b)):
                # Jaccard Distance, 0 is same, 1 is completely different.
                dist.append(self.jaccard_dist(a[i], b[j]))
            closest.append(np.argmin(dist))
            closest_dist.append(dist[np.argmin(dist)])
        return closest, closest_dist

    def mean_centroids(self, cluster, cluster_dist, b):
        new_centroids = []
        sorted_clusters = []
        cluster_length = []
        for i in range(len(b)):
            sorted_clusters.append([])
            for j in range(len(cluster)):
                if cluster[j] == i:
                    sorted_clusters[i].append(cluster_dist[j])
            # in case cluster has no points
            if sorted_clusters[i]:
                new_centroids.append(mean(sorted_clusters[i]))
            else:
                new_centroids.append(0)
            cluster_length.append(len(sorted_clusters[i]))
        return new_centroids, cluster_length

    # find new cluster point
    def find_new_centroid(self, new_centroid, a, dist):
        centroid = []
        index = []
        difference_arr = 1.0
        diff_ind = 0
        for i in range(len(new_centroid)):
            for j in range(len(a)):
                if difference_arr > np.absolute(new_centroid[i] - dist[j]):
                    difference_arr = np.absolute(new_centroid[i] - dist[j])
                    diff_ind = j
            if a[diff_ind].eq(pd.Series(centroid, dtype='object')).any():
                centroid.append(new_centroid[i])
            else:
                centroid.append(a[diff_ind])
            difference_arr = 1.0
        return centroid

    def sse(self, a, cluster):
        closest, closest_dist = self.find_closest(a, cluster)
        squared = np.square(closest_dist)
        sse = np.sum(squared)
        return sse

    # main train body
    def train_evaluate(self):
        for k in range(2, 7, 1):
            # getting random initial centroids
            init_centroids = random.sample(range(0, len(self.df)), k)

            a = []
            b = []
            # using lists
            for i in range(len(self.df)):
                a.append(self.df.loc[i])
            for j in init_centroids:
                b.append(self.df.loc[j])

            # initial clusters
            closest, closest_dist = self.find_closest(a, b)
            new_centroid, clusterlen = self.mean_centroids(closest, closest_dist, b)
            ncent = self.find_new_centroid(new_centroid, a, closest_dist)
            # short loop as it takes too long for convergence if allowed
            for i in range(4):
                prev_centroid = new_centroid
                closest, closest_dist = self.find_closest(a, ncent)
                new_centroid, clusterlen = self.mean_centroids(closest, closest_dist, b)
                ncent = self.find_new_centroid(new_centroid, a, closest_dist)
                # if algorithm converges
                if collections.Counter(prev_centroid) == collections.Counter(new_centroid):
                    break
            print("Value of K: ", k)
            print("Size of each cluster: ", clusterlen)
            # sse is sum of (distance between nearest cluster and each data point)^2
            print("SSE: ", self.sse(a, ncent))
        return 0


# start of python
if __name__ == '__main__':
    kmean = KMeans("https://ogunonu.github.io/usnewshealth.txt")
    kmean.preprocess()
    kmean.train_evaluate()
