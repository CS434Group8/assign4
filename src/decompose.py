import numpy as np
import copy


class PCA():
    """
    PCA. A class to reduce dimensions
    """

    def __init__(self, retain_ratio):
        """

        :param retain_ratio: percentage of the variance we maitain (see slide for definition)
        """
        self.retain_ratio = retain_ratio

    @staticmethod
    def mean(x):
        """
        returns mean of x
        :param x: matrix of shape (n, m)
        :return: mean of x of with shape (m,)
        """
        return x.mean(axis=0)

    @staticmethod
    def cov(x):
        """
        returns the covariance of x,
        :param x: input data of dim (n, m)
        :return: the covariance matrix of (m, m)
        """
        return np.cov(x.T)

    @staticmethod
    def eig(c):
        """
        returns the eigval and eigvec
        :param c: input matrix of dim (m, m)
        :return:
            eigval: a numpy vector of (m,)
            eigvec: a matrix of (m, m), column ``eigvec[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``eigval[i]``
            Note: eigval is not necessarily ordered
        """

        eigval, eigvec = np.linalg.eig(c)
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)
        return eigval, eigvec


    def fit(self, x):
        """
        fits the data x into the PCA. It results in self.eig_vecs and self.eig_values which will
        be used in the transform method
        :param x: input data of shape (n, m); n instances and m features
        :return:
            sets proper values for self.eig_vecs and eig_values
        """

        self.eig_vals = None
        self.eig_vecs = None

        x = x - PCA.mean(x)
        cov=PCA.cov(x)
        self.eig_vals,self.eig_vecs=PCA.eig(cov)
        
        # print(self.eig_vals)
        # print(self.eig_vals.shape)
        ########################################
        #       YOUR CODE GOES HERE            #
        ########################################

    def findD(self):
        totalEigVal=self.eig_vals.sum()
        goalEigval=totalEigVal*self.retain_ratio
        currentEigval=0
        index=0
        while(currentEigval<goalEigval):
            currentEigval+=self.eig_vals[index]
            index+=1
        print('D is : ',index)
        return index

    def transform(self, x):
        """
        projects x into lower dimension based on current eig_vals and eig_vecs
        :param x: input data of shape (n, m)
        :return: projected data with shape (n, len of eig_vals)
        """

        if isinstance(x, np.ndarray):
            x = np.asarray(x)
        if self.eig_vecs is not None:
            return np.matmul(x, self.eig_vecs)
        else:
            return x
