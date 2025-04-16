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



def filter_task_betas(task_betas, df_conditions, phase="delay", first_delay_only=True):
    """
    Filters the task betas based on specified conditions.

    Parameters:
    - task_betas (dict): Dictionary containing beta values for each task/session/run.
    - df_conditions (dict): Dictionary containing conditions for each task/session/run.
    - phase (str): The phase to filter on (default is "delay").
    - first_delay_only (bool): If True, filters only trials where 'prev_stimulus' is 1000.

    Returns:
    - filtered_task_betas (dict): Filtered beta values.
    - filtered_task_df (dict): Filtered conditions dataframe.
    """

    # Deep copy to avoid modifying original data
    filtered_task_betas = copy.deepcopy(task_betas)
    filtered_task_df = copy.deepcopy(df_conditions)

    for task, sessions in task_betas.items():
        for sess, runs in sessions.items():
            for run, betas in runs.items():
                task_df = df_conditions[task][sess].get(run)

                # Safety check to ensure data exists
                if task_df is None or betas is None:
                    print(f"Warning: Missing data for task={task}, session={sess}, run={run}")
                    continue

                # Apply filtering conditions
                if first_delay_only:
                    condition_mask = (task_df['prev_stimulus'] == 1000) & (task_df["regressor_type"] == phase)
                else:
                    condition_mask = task_df["regressor_type"] == phase

                # Ensure indexing consistency and type safety
                filtered_df = task_df[condition_mask].reset_index(drop=True)
                betas_phase = betas[:, condition_mask.to_numpy()]

                # Store filtered data
                filtered_task_betas[task][sess][run] = betas_phase
                filtered_task_df[task][sess][run] = filtered_df

    return filtered_task_betas, filtered_task_df


