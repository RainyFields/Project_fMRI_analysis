import numpy as np
import copy
import sklearn
from sklearn import svm
import sklearn.metrics
import sklearn.linear_model
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler


def decoding(trainset, testset, trainlabels, testlabels, classifier='distance', confusion=False,
             feature_normalization=False):
    """ distance/pearson based decoding is best for fMRI"""

    # Apply feature normalization if requested
    if feature_normalization:
        scaler = StandardScaler()
        trainset = scaler.fit_transform(trainset)  # Fit on trainset and transform
        testset = scaler.transform(testset)  # Transform testset based on trainset statistics

    unique_labels = np.unique(trainlabels)

    if classifier in ['distance', 'cosine']:
        #### Create prototypes from trainset
        prototypes = {}
        for label in unique_labels:
            ind = np.where(trainlabels == label)[0]
            prototypes[label] = np.mean(trainset[ind, :], axis=0)

        #### Now classify each sample in the testset
        predictions = []
        for i in range(testset.shape[0]):
            # Correlate sample with each prototype
            rs = []
            for label in prototypes:
                if classifier == 'distance':
                    rs.append(stats.pearsonr(prototypes[label], testset[i, :])[0])
                if classifier == 'cosine':
                    rs.append(np.dot(prototypes[label], testset[i, :]) /
                              (np.linalg.norm(prototypes[label]) * np.linalg.norm(testset[i, :])))

            # Find the closest prototype for sample
            max_ind = np.argmax(np.asarray(rs))
            predictions.append(unique_labels[max_ind])

        predictions = np.asarray(predictions)

    elif classifier == 'logistic':
        clf = sklearn.linear_model.LogisticRegression(solver='liblinear')
        clf.fit(trainset, trainlabels)
        predictions = clf.predict(testset)

    elif classifier == 'ridge':
        clf = sklearn.linear_model.RidgeClassifier(solver='svd', max_iter=1000)
        clf.fit(trainset, trainlabels)
        predictions = clf.predict(testset)

    elif classifier == 'svm':
        clf = svm.SVC(kernel='linear', probability=True)
        clf.fit(trainset, trainlabels)
        predictions = clf.predict(testset)

    accuracy = predictions == np.asarray(testlabels)
    confusion_mat = sklearn.metrics.confusion_matrix(testlabels, predictions, labels=unique_labels)

    if confusion:
        return accuracy, confusion_mat
    else:
        return accuracy



