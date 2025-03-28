import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
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

    return np.column_stack((y_test, y_pred)), balanced_accuracy_score(y_test, y_pred)


def cross_validation(X, y, modelType, band, multiclass, subjects=None):
    # Subject-Dependent task
    if subjects is None: 
        model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True))
        cv = StratifiedKFold(n_splits=5, shuffle=True)

        scoring = ['balanced_accuracy', 'roc_auc_ovr'] if multiclass else ['balanced_accuracy', 'roc_auc']
        acc_scores = cross_validate(model, X.reshape(X.shape[0], -1), y, cv=cv, scoring=scoring)
        balanced_acc = acc_scores['test_balanced_accuracy']
        roc_auc = acc_scores['test_' + scoring[1]]

    # Subject-Independent task
    else: 
        balanced_acc = []
        roc_auc = []

        unique_subjects = np.array(list(set(subjects)))
        y_threshold = {s: np.percentile(y[subjects == s], 50) for s in unique_subjects}
        y = (y > np.vectorize(y_threshold.get)(subjects)).astype(int)

        cv = KFold(n_splits=5, shuffle=True)

        for (train_idx, test_idx) in tqdm(cv.split(unique_subjects), total=5, desc=f"{band} Band"):

            train_subjects = unique_subjects[train_idx]
            test_subjects = unique_subjects[test_idx]

            mask_train = np.isin(subjects, train_subjects)
            mask_test = np.isin(subjects, test_subjects)

            # Extract features and labels
            X_train = X[mask_train].reshape(mask_train.sum(), -1)
            y_train = y[mask_train]
            X_test = X[mask_test].reshape(mask_test.sum(), -1)
            y_test = y[mask_test]

            model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            balanced_acc.append(balanced_accuracy_score(y_test, y_pred))
            roc_auc.append(roc_auc_score(y_test, y_pred))

    return round(np.mean(balanced_acc), 3), round(np.std(balanced_acc), 3), round(np.mean(roc_auc), 3), round(np.std(roc_auc), 3)


def subject_dependent_crossval(feature_matrix, targetType, modelType, multiclass, show_crossval):
    subject_acc_scores = []
    subject_roc_scores = []

    subjects = set(sample['subject'] for sample in feature_matrix)

    for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'All']):
        acc_scores_valence = []
        roc_scores_valence = []
        acc_scores_arousal = []
        roc_scores_arousal = []
        acc_scores_target = []
        roc_scores_target = []

        for subject in tqdm(subjects, desc=f"{band} Band"):
            subject_data = [sample for sample in feature_matrix if sample['subject'] == subject]
            X = np.array([sample['feat'][i] for sample in subject_data])
           
            if targetType == "valence_arousal":
                y_valence = np.array([sample['label'][0] for sample in subject_data])
                y_valence = (y_valence > np.percentile(y_valence, 50)).astype(int)
                
                y_arousal = np.array([sample['label'][1] for sample in subject_data])
                y_arousal = (y_arousal > np.percentile(y_arousal, 50)).astype(int)

                (mean_score_valence, std_score_valence, 
                mean_roc_valence, std_roc_valence) = cross_validation(X, y_valence, modelType, "Valence "+band, multiclass)
                (mean_score_arousal, std_score_arousal, 
                mean_roc_arousal, std_roc_arousal) = cross_validation(X, y_arousal, modelType, "Arousal "+band, multiclass)

                acc_scores_valence.append(mean_score_valence)
                roc_scores_valence.append(mean_roc_valence)
                acc_scores_arousal.append(mean_score_arousal)
                roc_scores_arousal.append(mean_roc_arousal)

                if show_crossval:
                    print(f" -- Subject {subject} - 5-Fold Cross Validation Valence|Arousal --\n"
                          f"\tBalanced Accuracy ± Std: "
                          f"{np.round(mean_score_valence, 3)} ± {np.round(std_score_valence, 3)} | "
                          f"{np.round(mean_score_arousal, 3)} ± {np.round(std_score_arousal, 3)}\n"
                          f"\tROC-AUC ± Std: "
                          f"{np.round(mean_roc_valence, 3)} ± {np.round(std_roc_valence, 3)} | "
                          f"{np.round(mean_roc_arousal, 3)} ± {np.round(std_roc_arousal, 3)}")
        
            elif targetType == "single":
                y = np.array([sample['label'] for sample in subject_data])
                mean_score, std_score, mean_roc, std_roc = cross_validation(X, y, modelType, band, multiclass)
                acc_scores_target.append(mean_score)
                roc_scores_target.append(mean_roc)

                if show_crossval:
                    print(f"-- Subject {subject} - 5-Fold Cross Validation --\n"
                          f"\tBalanced Accuracy ± Std: "
                          f"{np.round(mean_score, 3)} ± {np.round(std_score, 3)}\n"
                          f"\tROC-AUC ± Std: "
                          f"{np.round(mean_roc, 3)} ± {np.round(std_roc, 3)}")
            
            else:
                raise ValueError(f"Invalid targetType: {targetType}. Choose from 'single' or 'valence_arousal'.")

        if targetType == "valence_arousal":
            # Average subject-dependent balanced accuracy across all subjects
            subject_acc_scores.append([round(np.mean(acc_scores_valence), 3), round(np.mean(acc_scores_arousal), 3)])
            subject_roc_scores.append([round(np.mean(roc_scores_valence), 3), round(np.mean(roc_scores_arousal), 3)])
        else:
            # Average subject-dependent balanced across all subjects
            subject_acc_scores.append(round(np.mean(acc_scores_target), 3))
            subject_roc_scores.append(round(np.mean(roc_scores_target), 3))
    
    return subject_acc_scores, subject_roc_scores


def subject_independent_crossval(dataset, targetType, modelType, multiclass, show_crossval):
    acc_scores_target = []
    roc_scores_target = []
    
    for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'All']):
        subjects = np.array([sample['subject'] for sample in dataset])
        X = np.array([sample['feat'][i] for sample in dataset])

        if targetType == "valence_arousal":
            y_valence = np.array([sample['label'][0] for sample in dataset])
            y_arousal = np.array([sample['label'][1] for sample in dataset])

            (mean_score_valence, std_score_valence, 
             mean_roc_valence, std_roc_valence) = cross_validation(X, y_valence, modelType, 
                                                                     "Valence "+band, multiclass, subjects)
            (mean_score_arousal, std_score_arousal, 
             mean_roc_arousal, std_roc_arousal) = cross_validation(X, y_arousal, modelType,
                                                                     "Arousal "+band, multiclass, subjects)

            acc_scores_target.append([mean_score_valence, mean_score_arousal])
            roc_scores_target.append([mean_roc_valence, mean_roc_arousal])

            if show_crossval:
                print(f"-- 5-Fold Cross Validation Valence|Arousal --\n"
                      f"\tBalanced Accuracy ± Std: "
                      f"{np.round(mean_score_valence, 3)} ± {np.round(std_score_valence, 3)} | "
                      f"{np.round(mean_score_arousal, 3)} ± {np.round(std_score_arousal, 3)}\n"
                      f"\tROC-AUC ± Std: "
                      f"{np.round(mean_roc_valence, 3)} ± {np.round(std_roc_valence, 3)} | "
                      f"{np.round(mean_roc_arousal, 3)} ± {np.round(std_roc_arousal, 3)}")
        
        elif targetType == "single":
            y = np.array([sample['label'] for sample in dataset])

            mean_score, std_score, mean_roc, std_roc = cross_validation(X, y, modelType, band, multiclass, subjects)

            acc_scores_target.append(mean_score)
            roc_scores_target.append(mean_roc)

            if show_crossval:
                print(f"-- 5-Fold Cross Validation --\n"
                      f"\tBalanced Accuracy ± Std: "
                      f"{np.round(mean_score, 3)} ± {np.round(std_score, 3)}\n"
                      f"\tROC-AUC ± Std: "
                      f"{np.round(mean_roc, 3)} ± {np.round(std_roc, 3)}")
        
        else:
            raise ValueError(f"Invalid targetType: {targetType}. Choose from 'single' or 'valence_arousal'.")
    
    return acc_scores_target, roc_scores_target
        


def crossval_classifier(feature_matrix, task, targetType, modelType, multiclass=False, show_crossval=False):
    if task == "subject_dependent":
        return subject_dependent_crossval(feature_matrix, targetType, modelType, multiclass, show_crossval)
    elif task == "subject_independent":
        return subject_independent_crossval(feature_matrix, targetType, modelType, multiclass, show_crossval)
    else:
        raise ValueError(f"Invalid task: {task}. Choose from 'subject_dependent' or 'subject_independent'.")
