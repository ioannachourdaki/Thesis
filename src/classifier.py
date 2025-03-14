import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return np.column_stack((y_test, y_pred)), accuracy_score(y_test, y_pred)
