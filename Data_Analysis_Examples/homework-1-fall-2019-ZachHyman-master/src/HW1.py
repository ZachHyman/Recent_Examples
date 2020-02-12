import matplotlib.pyplot as plt
import numpy as np
from folium.plugins import HeatMap
from matplotlib import patches
import sklearn
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import csv
import folium
from mpl_toolkits import mplot3d
import matplotlib.lines as lines
import pandas as pd

from scipy import linalg
import itertools
global meanVal
global maxVal
global minVal

#POINT OBJECT METHOD DECLARATION
class Point:
    def __init__(self, lat=0.0, lon=0.0, val=0, group=0):
        self.lat, self.lon, self.val, self.group = lat, lon, val, group
    def __str__(self):
        return "Point Values: Lat " + str(self.lat) + " Lon " + str(self.lon) + " Val " + str(self.val) + " Group " + str(self.group)

points = []


#EXTRACT AND REFORMAT DATA
with open('nyc_listings.csv', newline='') as f:
    reader = csv.reader(f)
    i = 0
    coords = []

    minVal = 100000000
    maxVal = 0
    sumVal = 0

    for row in reader:
        # Skip Header Row
        i += 1
        if i == 1:
            continue

        formVal = float(row[60].replace(',', '')[1:])

        # Setup for Mean normilization
        if formVal < minVal:
            minVal = formVal
        if formVal > maxVal:
            maxVal = formVal
        sumVal += formVal

        coords.append([float(row[48]), float(row[49]), formVal])

        # if (i == 5000):
        #     break

    meanVal = sumVal / len(coords)

    #NORMALIZE DATA
    for entry in coords:
        #LIST ARRAY METHOD
        entry[2] = ((entry[2] - meanVal) / (maxVal - minVal))

        #OBJECT ARRAY METHOD
        #points.append(Point(entry[0],entry[1],((entry[2] - meanVal) / (maxVal - minVal)) ))

    #VALUE STATS
    # print(meanVal)
    # print(maxVal)
    # print(minVal)

############################################################

    #CREATE K MEANS MODEL
    X = np.array(coords)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)


    #2D MODEL K MEANS
    # plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
    # plt.show()

    #3D MODEL K MEANS
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels_, cmap='rainbow');
    # plt.show()


    #UN-NORMALIZING PRICES AND GETTING AVERAGES
    c = np.column_stack((X, kmeans.labels_))

    # zeroCount = 0
    # zeroSum = 0
    # oneCount = 0
    # oneSum = 0
    # twoCount = 0
    # twoSum = 0
    # threeCount = 0
    # threeSum = 0
    #
    # for elem in c:
    #     if elem[3] == 0.0:
    #         zeroCount +=1
    #         zeroSum += elem[2]
    #     if elem[3] == 1.0:
    #         oneCount +=1
    #         oneSum += elem[2]
    #     if elem[3] == 2.0:
    #         twoCount +=1
    #         twoSum += elem[2]
    #     if elem[3] == 3.0:
    #         threeCount +=1
    #         threeSum += elem[2]
    #
    # print("Average Price in Cluster 0: " + str(  (zeroSum/zeroCount) * (maxVal - minVal) + meanVal))
    # print("Average Price in Cluster 1: " + str(  (oneSum/oneCount) * (maxVal - minVal) + meanVal))
    # print("Average Price in Cluster 2: " + str(  (twoSum/twoCount) * (maxVal - minVal) + meanVal))
    # print("Average Price in Cluster 3: " + str(  (threeSum/threeCount) * (maxVal - minVal) + meanVal))
    # print(kmeans.cluster_centers_)


    #LOSS PLOT
    #print(kmeans.inertia_)
    # lossArray = np.array([[1, 279.4384108211212],
    #                       [2, 185.12375300817536],
    #                       [3, 137.23219597573416],
    #                       [4, 105.5788530698801],
    #                       [5, 88.5757336660534],
    #                       [6, 74.89244127729593],
    #                       [7, 66.22902334737653],
    #                       [8, 59.18097985803997],
    #                       [9, 52.46920471076173],
    #                       [10, 47.87907618286269]])
    # plt.plot(lossArray[:, 0], lossArray[:, 1])
    # plt.plot([1, 10], [279.4384108211212, 47.87907618286269 ])
    # plt.show()
############################################################

    #CREATE HEIR CLUSTER MODEL
    # X = np.array(coords)
    # cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    # cluster.fit_predict(X)
    #
    # #2D MODEL HEIR
    # plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='rainbow')
    # plt.show()
    #
    # # 3D MODEL HEIR
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=cluster.labels_, cmap='rainbow');
    # plt.show()

    #HEIR SIHLOUETTE DATA
    # print(sklearn.metrics.silhouette_score(X, cluster.labels_, metric='euclidean'))
    # lossArray = np.array([[2, 0.36491387532534497],
    #                       [3, 0.33072476614527047],
    #                       [4, 0.3521486648295497],
    #                       [5, 0.35762471426046866],
    #                       [6, 0.3137026275816868],
    #                       [7, 0.3303704379581293],
    #                       [8, 0.3316269758431951],
    #                       [9, 0.35391034291260337],
    #                       [10, 0.3583846500347575]])
    # plt.plot(lossArray[:, 0], lossArray[:, 1])
    # # plt.plot([1, 10], [279.4384108211212, 47.87907618286269 ])
    # plt.show()
###############################################################


    #3D MATLIB PLOT ATTEMPT 1
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # progress = 0
    # for p in points:
    #     if progress % 5000 == 0:
    #         print("Progress: " + str(progress))
    #     ax.scatter3D(p.lat, p.lon, p.val);
    #     progress += 1
    #
    # plt.show()

##################################################################
# X = np.array(coords)
#
# def generateBaseMap(default_location = [40.693943, -73.985880]):
#     base_map = folium.Map(location=default_location )
#     return base_map
# base_map = generateBaseMap()
#
# HeatMap(X,radius=8, max_zoom=13).add_to(base_map)
# base_map.save('index.html')
####################################################################

#BASEMAP GENERATION WITH CLUSTERED POINTS

def generateBaseMap(default_location = [40.693943, -73.985880]):
    base_map = folium.Map(location=default_location )
    return base_map

base_map = generateBaseMap()


for elem in c:
    if elem[3] == 0.0:
        folium.CircleMarker( location=[ elem[0], elem[1] ], fill=True, color='#ff0000', radius=2 ).add_to( base_map )
    if elem[3] == 1.0:
        folium.CircleMarker( location=[ elem[0], elem[1] ], fill=True, color='#0000ff', radius=2 ).add_to( base_map )
    if elem[3] == 2.0:
        folium.CircleMarker( location=[ elem[0], elem[1] ], fill=True, color='#008000', radius=2 ).add_to( base_map )
    if elem[3] == 3.0:
        folium.CircleMarker( location=[ elem[0], elem[1] ], fill=True, color='#ffff00', radius=2 ).add_to( base_map )



base_map.save('index.html')
