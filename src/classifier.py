import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)


def classifier(X, y, modelType, test_size=0.2):
    N = X.shape[0]
    X_reshaped = X.reshape(N, -1) 

    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=test_size, random_state=42)

    if modelType == "svm":
        model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))

    elif modelType == "knn":
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

    elif modelType == "quadratic":
        model = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())
    
    else:
        raise ValueError(f"Invalid modelType: {modelType}. Choose from 'svm', 'knn', or 'quadratic'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return np.column_stack((y_test, y_pred)), accuracy_score(y_test, y_pred)


def cross_validation(X, y, modelType, band, subjects=None):
    if modelType == "svm":
        model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))

    elif modelType == "knn":
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

    elif modelType == "quadratic":
        model = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())
    
    else:
        raise ValueError(f"Invalid modelType: {modelType}. Choose from 'svm', 'knn', or 'quadratic'.")

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # Perform 5-fold cross-validation
    if subjects is None: # Subject-Dependent task
        for _ in tqdm(range(1), desc=f"{band} Band 5-Fold Cross Validation"):
            cv_scores = cross_val_score(model, X.reshape(X.shape[0], -1), y, cv=cv, scoring='accuracy')

    else: # Subject-Independent task
        cv_scores = []
        unique_subjects = np.unique(subjects)

        for (train_idx, test_idx) in tqdm(skf.split(unique_subjects, [y[subjects == s][0] for s in unique_subjects]),
                                                total=5, desc=f"{band} Band 5-Fold Cross Validation"):

            train_subjects = unique_subjects[train_idx]
            test_subjects = unique_subjects[test_idx]

            mask_train = np.isin(subjects, train_subjects)
            mask_test = np.isin(subjects, test_subjects)

            # Extract features and labels
            X_train = X[mask_train].reshape(mask_train.sum(), -1)
            y_train = y[mask_train]
            X_test = X[mask_test].reshape(mask_test.sum(), -1)
            y_test = y[mask_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores.append(accuracy)

    return round(np.mean(cv_scores), 3), round(np.var(cv_scores), 3)


def subject_dependent_crossval(feature_matrix, targetType, modelType, show_crossval):
    scores_valence = []
    scores_arousal = []
    scores_target = []
    subject_scores = []

    subjects = set(sample['subject'] for sample in feature_matrix)

    for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'All']):
        for subject in subjects:
            subject_data = [sample for sample in feature_matrix if sample['subject'] == subject]
            X = np.array([sample['feat'][i] for sample in subject_data])
           
            if targetType == "valence_arousal":
                y_valence = (np.array([sample['label'][0] for sample in subject_data]) > 5).astype(int)
                y_arousal = (np.array([sample['label'][1] for sample in subject_data]) > 5).astype(int)

                mean_score_valence, var_score_valence = cross_validation(X, y_valence, modelType, band+" Valence")
                mean_score_arousal, var_score_arousal = cross_validation(X, y_arousal, modelType, band+" Arousal")

                scores_valence.append(mean_score_valence)
                scores_arousal.append(mean_score_arousal)

                if show_crossval:
                    print(f"Subject {subject} - 5-Fold Valence-Arousal Accuracy and Varience: "
                        f"{np.round(mean_score_valence, 3)}-{np.round(mean_score_arousal, 3)}, "
                        f"{np.round(var_score_valence, 3)}-{np.round(var_score_arousal, 3)}")
        
            elif targetType == "single":
                y = np.array([sample['label'] for sample in subject_data])
                mean_score, var_score = cross_validation(X, y, modelType, band)
                scores_target.append(mean_score)

                if show_crossval:
                    print(f"Subject {subject} - 5-Fold Accuracy and Varience: "
                        f"{np.round(mean_score, 3)}, {np.round(var_score, 3)}")
            
            else:
                raise ValueError(f"Invalid targetType: {targetType}. Choose from 'single' or 'valence_arousal'.")

        if targetType == "valence_arousal":
            # Average subject-dependent accuracy across all subjects
            subject_scores.append([round(np.mean(scores_valence), 3), round(np.mean(scores_arousal), 3)])
        else:
            # Average subject-dependent accuracy across all subjects
            subject_scores.append(round(np.mean(scores_target), 3))
    
    return subject_scores


def subject_independent_crossval(dataset, targetType, modelType, show_crossval):

    scores_target = []

    subjects = np.array([sample['subject'] for sample in dataset])
    
    for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'All']):
        X = np.array([sample['feat'][i] for sample in dataset])

        if targetType == "valence_arousal":
            labels_valence = (np.array([sample['label'][0] for sample in dataset]) > 5).astype(int)
            labels_arousal = (np.array([sample['label'][1] for sample in dataset]) > 5).astype(int)

            mean_score_valence, var_score_valence = cross_validation(X, labels_valence, modelType, band+" Valence", subjects)
            mean_score_arousal, var_score_arousal = cross_validation(X, labels_arousal, modelType, band+" Arousal", subjects)

            scores_target.append([mean_score_valence, mean_score_arousal])

            if show_crossval:
                print(f"5-Fold Valence-Arousal Accuracy and Varience: "
                    f"{mean_score_valence}-{mean_score_arousal}, "
                    f"{var_score_valence}-{var_score_arousal}")
        
        elif targetType == "single":
            labels = np.array([sample['label'] for sample in dataset])

            mean_score, var_score = cross_validation(X, labels, modelType, band, subjects)

            scores_target.append(mean_score)

            if show_crossval:
                print(f"5-Fold Accuracy and Varience: "
                    f"{mean_score}-{mean_score}")
        
        else:
            raise ValueError(f"Invalid targetType: {targetType}. Choose from 'single' or 'valence_arousal'.")
    
    return scores_target
        


def crossval_classifier(feature_matrix, task, targetType, modelType, show_crossval=False):
    if task == "subject_dependent":
        return subject_dependent_crossval(feature_matrix, targetType, modelType, show_crossval)
    elif task == "subject_independent":
        return subject_independent_crossval(feature_matrix, targetType, modelType, show_crossval)
    else:
        raise ValueError(f"Invalid task: {task}. Choose from 'subject_dependent' or 'subject_independent'.")
