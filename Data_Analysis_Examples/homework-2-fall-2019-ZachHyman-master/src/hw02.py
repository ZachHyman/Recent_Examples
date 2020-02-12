import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import time



arrayX = np.load('mnist_data.npy')
labelsY = np.load('mnist_labels.npy')

# print(arrayX.shape)
# print(labelsY.shape)

#PROBLEM 1.a
X_train, X_test, y_train, y_test = train_test_split(arrayX, labelsY, test_size=0.2, random_state=0)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

#PROBLEM 1.b
def callLogisticRegression():
    iter = 300
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=iter).fit(X_train,y_train)
    time.sleep(5)
    print("Logistic Regression with " + str(iter) + " iterations" )
    print("\tTraining set accuracy " + str(clf.score(X_train, y_train)))
    print("\tTest set accuracy " + str(clf.score(X_test, y_test)))
    return

#callLogisticRegression()


#PROBLEM 1.c
def callKNN(k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    print("KNN with k = " + str(k))
    print("\tTraining set accuracy " + str(neigh.score(X_train, y_train)))
    print("\tTest set accuracy " + str(neigh.score(X_test, y_test)))
    return

#callKNN(23)

KNNresults = [
    [1, 1.0, 0.9669047619047619],
    [3, 0.980297619047619, 0.96],
    [5, 0.9732142857142857, 0.9616666666666667],
    [7, 0.9686904761904762, 0.959047619047619],
    [9, 0.9627976190476191, 0.955952380952381],
    [11,0.9588095238095238,0.9545238095238096],
    [13,0.957202380952381,0.9530952380952381],
    [15,0.9548214285714286,0.9511904761904761],
    [17,0.9524404761904762,0.9502380952380952],
    [19,0.9498214285714286,0.9483333333333334],
    [21,0.9469642857142857,0.9471428571428572]
]

def column(matrix, i):
    return [row[i] for row in matrix]

def plotKNN():

    x = np.arange(len(column(KNNresults,0)))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, column(KNNresults,1), width, label='Train')
    rects2 = ax.bar(x + width/2, column(KNNresults,2), width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Size of K')
    ax.set_xticks(x)
    ax.set_xticklabels(column(KNNresults,0))
    ax.legend()
    plt.ylim([0.94, 1.00])
    fig.tight_layout()
    plt.show()

plotKNN()

#PROBLEM 2.a & 2.b
def callPCA():
    pca = PCA()
    pca.fit(X_train)
    # PROBLEM 2.a
    print(pca.explained_variance_ratio_)
    # PROBLEM 2.b
    CDF = np.cumsum(pca.explained_variance_ratio_)
    #print(CDF)
    print("CDF of 200 features " + str(CDF[199]))
    print("CDF of 300 features " + str(CDF[299]))
    print("CDF of 375 features " + str(CDF[374]))
    plt.plot(CDF)
    plt.xlabel('Number of Principle Components')
    plt.ylabel('Variance Explained')
    plt.title('CDF of Explained Variance Ratios')
    plt.grid()
    plt.show()


#callPCA()

#PROBLEM 2.c
def callKNNClassifier():
    clf = make_pipeline(PCA(n_components=375), KNeighborsClassifier(n_neighbors=1))
    clf.fit(X_train, y_train)
    print("\tTraining set accuracy " + str(clf.score(X_train, y_train)))
    print("\tTest set accuracy " + str(clf.score(X_test, y_test)))
    return


#callKNNClassifier()

#PROBLEM 2.d
def KNNClassicTime(sampleSize):
    if sampleSize == 21000:
        neigh = KNeighborsClassifier(n_neighbors=1)
        start_time = time.time()
        neigh.fit(arrayX, labelsY)
        return time.time() - start_time

    X_train, X_test, y_train, y_test = train_test_split(arrayX, labelsY, test_size=1-(sampleSize/21000),random_state=1234)
    neigh = KNeighborsClassifier(n_neighbors=1)

    start_time = time.time()
    neigh.fit(X_train, y_train)
    return time.time() - start_time


#print("--- %s seconds ---" % (KNNClassicTime(21000)))

KNNClassic = [
    [3000,0.1662290096282959],
    [6000,0.6316862106323242],
    [9000,1.4704461097717285],
    [12000,2.0542666912078857],
    [15000,3.1238210201263428],
    [18000,4.1087751388549805],
    [21000,6.212893009185791]
]

def KNNPCATime(sampleSize,componentNumber):

    if sampleSize == 21000:
        clf = make_pipeline(PCA(n_components=componentNumber), KNeighborsClassifier(n_neighbors=1))
        start_time = time.time()
        clf.fit(arrayX, labelsY)
        return time.time() - start_time

    X_train, X_test, y_train, y_test = train_test_split(arrayX, labelsY, test_size=1-(sampleSize / 21000),random_state=1234)
    clf = make_pipeline(PCA(n_components=componentNumber), KNeighborsClassifier(n_neighbors=1))

    start_time = time.time()
    clf.fit(X_train, y_train)
    return time.time() - start_time

#print("--- %s seconds ---" % (KNNPCATime(21000,750)))

KNNPCA_50 = [
    [3000,0.5054588317871094],
    [6000,0.7227740287780762],
    [9000,1.0335280895233154],
    [12000,1.5438511371612549],
    [15000,1.7227141857147217],
    [18000,2.1094279289245605],
    [21000,2.206686019897461]
]

KNNPCA_150 = [
    [3000,0.4743337631225586],
    [6000,0.9882810115814209],
    [9000,1.5087990760803223],
    [12000,2.07521915435791],
    [15000,2.6514532566070557],
    [18000,3.210338830947876],
    [21000,3.7975800037384033]
]
KNNPCA_250 = [
    [3000,0.9654428958892822],
    [6000,1.7333648204803467],
    [9000,2.4914181232452393],
    [12000,3.3316450119018555],
    [15000,4.173448085784912],
    [18000,4.684069871902466],
    [21000,5.32666277885437]
]
KNNPCA_350 = [
    [3000,1.3677890300750732],
    [6000,2.2146430015563965],
    [9000,3.529472827911377],
    [12000,4.2175819873809814],
    [15000,5.351502180099487],
    [18000,5.971056938171387],
    [21000,9.59891414642334]
]
KNNPCA_450 = [
    [3000,1.8397088050842285],
    [6000,3.334534168243408],
    [9000,4.129649877548218],
    [12000,5.8114330768585205],
    [15000,7.034854888916016],
    [18000,9.744870901107788],
    [21000,12.18409776687622]
]
KNNPCA_550 = [
    [3000,2.144892930984497],
    [6000,4.03105092048645],
    [9000,5.48354697227478],
    [12000,8.075253009796143],
    [15000,9.202515840530396],
    [18000,11.064366817474365],
    [21000,12.68863821029663]
]
KNNPCA_650 = [
    [3000,0.9364371299743652],
    [6000,1.8270819187164307],
    [9000,2.682889938354492],
    [12000,3.849879741668701],
    [15000,5.170312881469727],
    [18000,5.796039819717407],
    [21000,7.008054971694946]
]
KNNPCA_750 = [
    [3000,0.9080469608306885],
    [6000,1.8872079849243164],
    [9000,2.6627988815307617],
    [12000,4.382929801940918],
    [15000,4.716165781021118],
    [18000,7.2001659870147705],
    [21000,8.068990230560303]
]


def plotKNNCompare():

    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(len(column(KNNClassic,0))))
    ax.set_xticklabels(column(KNNClassic,0))

    plt.plot(column(KNNClassic,1),label = "KNN Classic")
    plt.plot(column(KNNPCA_50,1),label = "KNN PCA 50")
    plt.plot(column(KNNPCA_150,1),label = "KNN PCA 150")
    plt.plot(column(KNNPCA_250, 1), label="KNN PCA 250")
    plt.plot(column(KNNPCA_350, 1), label="KNN PCA 350")
    plt.plot(column(KNNPCA_450, 1), label="KNN PCA 450")
    plt.plot(column(KNNPCA_550, 1), label="KNN PCA 550")
    plt.plot(column(KNNPCA_650, 1), label="KNN PCA 650")
    plt.plot(column(KNNPCA_750, 1), label="KNN PCA 750")

    plt.title("KNN Classifier models against Runtime", fontsize=16, fontweight='bold')
    plt.xlabel("Size of training set")
    plt.ylabel("Time")

    #plt.legend(loc='lower left')
    plt.legend()
    plt.show()

plotKNNCompare()