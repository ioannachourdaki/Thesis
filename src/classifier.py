import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score


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


def cross_validation(X, y, modelType, groups=None):
    N = X.shape[0]
    X_reshaped = X.reshape(N, -1)

    if modelType == "svm":
        model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))

    elif modelType == "knn":
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

    elif modelType == "quadratic":
        model = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())
    
    else:
        raise ValueError(f"Invalid modelType: {modelType}. Choose from 'svm', 'knn', or 'quadratic'.")

    # Perform 5-fold cross-validation
    if groups == None: # Subject-Dependent task
        cv = StratifiedKFold(n_splits=5, shuffle=True)
    else: # Subject-Independent task
        cv = GroupKFold(n_splits=5)
    
    cv_scores = cross_val_score(model, X_reshaped, y, cv=cv, groups=groups, scoring='accuracy')

    return np.mean(cv_scores), np.var(cv_scores)


def subject_dependent_crossval(feature_matrix, targetType, modelType, show_crossval):
    scores_valence = []
    scores_arousal = []
    scores_target = []
    subject_scores = []

    subjects = set(sample['subject'] for sample in feature_matrix)

    for band in range(6):
        for subject in subjects:
            subject_data = [sample for sample in feature_matrix if sample['subject'] == subject]
            X = np.array([sample['feat'][band] for sample in subject_data])
           
            if targetType == "valence_arousal":
                y_valence = (np.array([sample['label'][0] for sample in subject_data]) > 5).astype(int)
                y_arousal = (np.array([sample['label'][1] for sample in subject_data]) > 5).astype(int) 

                mean_score_valence, var_score_valence = cross_validation(X, y_valence, modelType)
                mean_score_arousal, var_score_arousal = cross_validation(X, y_arousal, modelType)

                scores_valence.append(mean_score_valence)
                scores_arousal.append(mean_score_arousal)

                if show_crossval:
                    print(f"Subject {subject} - 5-Fold Valence-Arousal Accuracy and Varience: "
                        f"{np.round(mean_score_valence, 3)}-{np.round(mean_score_arousal, 3)}, "
                        f"{np.round(var_score_valence, 3)}-{np.round(var_score_arousal, 3)}")
        
            elif targetType == "single":
                y = np.array([sample['label'] for sample in subject_data])
                mean_score, var_score = cross_validation(X, y, modelType)
                scores_target.append(mean_score)

                if show_crossval:
                    print(f"Subject {subject} - 5-Fold Accuracy and Varience: "
                        f"{np.round(mean_score, 3)}, {np.round(var_score, 3)}")
            
            else:
                raise ValueError(f"Invalid targetType: {targetType}. Choose from 'single' or 'valence_arousal'.")

        if targetType == "valence_arousal":
            # Average subject-dependent accuracy across all subjects
            subject_scores.append([np.mean(scores_valence), np.mean(scores_arousal)])
        else:
            # Average subject-dependent accuracy across all subjects
            subject_scores.append(np.mean(scores_target))
    
    return subject_scores
        # print(f"\nAverage Subject-Dependent Valence-Arousal Accuracy: "
            # f"{np.round(avg_score_valence * 100, 3)}%-{np.round(avg_score_arousal * 100, 3)}%")


def subject_independent_crossval(feature_matrix, targetType, modelType, show_crossval):
    scores_target = []

    groups = np.array([sample['subject'] for sample in feature_matrix])

    for band in range(6):
        X = np.array([sample['feat'][band] for sample in subject_data])

        if targetType == "valence_arousal":
            y_valence = (np.array([sample['label'][0] for sample in subject_data]) > 5).astype(int)
            y_arousal = (np.array([sample['label'][1] for sample in subject_data]) > 5).astype(int) 

            mean_score_valence, var_score_valence = cross_validation(X, y_valence, modelType, groups)
            mean_score_arousal, var_score_arousal = cross_validation(X, y_arousal, modelType, groups)

            scores_target.append([mean_score_valence, mean_score_arousal])

            if show_crossval:
                print(f"5-Fold Valence-Arousal Accuracy and Varience: "
                    f"{np.round(mean_score_valence, 3)}-{np.round(mean_score_arousal, 3)}, "
                    f"{np.round(var_score_valence, 3)}-{np.round(var_score_arousal, 3)}")
        
        elif targetType == "single":
            y = np.array([sample['label'] for sample in subject_data])
            mean_score, var_score = cross_validation(X, y, modelType, groups)
            scores_target.append(mean_score)

            if show_crossval:
                print(f"5-Fold Accuracy and Varience: "
                    f"{np.round(mean_score, 3)}, {np.round(var_score, 3)}")
        
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
