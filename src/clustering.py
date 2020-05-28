import numpy as np
import random

class KMeans():
    """
    KMeans. Class for building an unsupervised clustering model
    """

    def __init__(self, k, max_iter=20):

        """
        :param k: the number of clusters
        :param max_iter: maximum number of iterations
        """

        self.k = k
        self.max_iter = max_iter

    def init_center(self, x):
        """
        initializes the center of the clusters using the given input
        :param x: input of shape (n, m)
        :return: updates the self.centers
        """

        self.centers = np.zeros((self.k, x.shape[1]))


        datasize=len(x)
        selected=[]
        
        i=0
        while i<self.k:
            rand=random.randint(0, datasize-1)
            if(rand not in selected):
                selected.append(rand)
                i+=1
        
        for index in range(self.centers.shape[0]):
            self.centers[index]=x[selected[index]]
        ################################
        #      YOUR CODE GOES HERE     #
        ################################

    def revise_centers(self, x, labels):
        """
        it updates the centers based on the labels
        :param x: the input data of (n, m)
        :param labels: the labels of (n, ). Each labels[i] is the cluster index of sample x[i]
        :return: updates the self.centers
        """

        for i in range(self.k):
            wherei = np.squeeze(np.argwhere(labels == i), axis=1)
            self.centers[i, :] = x[wherei, :].mean(0)

    def predict(self, x):
        """
        returns the labels of the input x based on the current self.centers
        :param x: input of (n, m)
        :return: labels of (n,). Each labels[i] is the cluster index for sample x[i]
        """
        labels = np.zeros((x.shape[0]), dtype=int)
        datasetLen=x.shape[0]
        miniResult=np.full(datasetLen,np.inf)
        # print(miniResult.shape)
        centersCopy=self.centers.copy()
        
        centerIndex=0
        for center in centersCopy:
            centerMatrix=np.tile(center,(datasetLen,1))
            result=np.square((x-centerMatrix))
            result=np.sum(result,axis=1)            
            miniResult=np.minimum(miniResult,result)
            # print(miniResult)
            for i in range(miniResult.shape[0]):
                if(miniResult[i]==result[i]):
                    labels[i]=centerIndex
            # print(labels)
            centerIndex+=1
        # for index in range(datasetLen):
        #     data=x[index]
        #     print(data.shape)    

        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        return labels

    def get_sse(self, x, labels):
        """
        for a given input x and its cluster labels, it computes the sse with respect to self.centers
        :param x:  input of (n, m)
        :param labels: label of (n,)
        :return: float scalar of sse
        """

        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################

        sse = 0.
        clusterCenters=x.copy()
        index=0
        for label in labels:
            clusterCenters[index]=self.centers[label]
            index+=1
            
        diff=np.square(x-clusterCenters)
        sse=round(diff.sum(),3)
        
        # print('sse:',sse)
        
        return sse

    def get_purity(self, x, y):
        """
        computes the purity of the labels (predictions) given on x by the model
        :param x: the input of (n, m)
        :param y: the ground truth class labels
        :return:
        """
        labels = self.predict(x)
        purity = 0
        correct=0
        for index in range(len(self.centers)):
            clusteIndex=np.squeeze(np.argwhere(labels==index),axis=1)
            
            cluster=y[clusteIndex]
            occurances = np.bincount(cluster)
            maxIndex=np.argmax(occurances)
            maxOccurance=occurances[maxIndex]
            correct+=maxOccurance
        
        
        purity=round(correct/len(y),3)
        print('purity:',purity)
        # quit()
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        return purity

    def fit(self, x):
        """
        this function iteratively fits data x into k-means model. The result of the iteration is the cluster centers.
        :param x: input data of (n, m)
        :return: computes self.centers. It also returns sse_veersus_iterations for x.
        """

        # intialize self.centers
        self.init_center(x)

        sse_vs_iter = []
        for iter in range(self.max_iter):
            # finds the cluster index for each x[i] based on the current centers
            labels = self.predict(x)

            # revises the values of self.centers based on the x and current labels
            self.revise_centers(x, labels)

            # computes the sse based on the current labels and centers.
            sse = self.get_sse(x, labels)

            sse_vs_iter.append(sse)

        return sse_vs_iter