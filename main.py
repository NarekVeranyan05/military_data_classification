import numpy as np
import pandas as pd
import imageio.v3 as iio
from sklearn.svm import SVC
from scipy.stats import uniform, randint
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def readData(dirs = [], class_part_index = 2):
    img_paths = [Path(d) for d in dirs]
    paths = []
    for path in img_paths:
        paths += list(path.glob('*.jpeg'))
        paths += list(path.glob('*.jpg'))
        paths += list(path.glob('*.png'))

    X = pd.DataFrame([iio.imread(path).flatten() for path in paths])
    y = pd.Series([path.parts[class_part_index] for path in paths])
    return X, y

def getModel():
    pipe = Pipeline([
        ('scaler', StandardScaler()),   # zero mean, unit variance
        ('svc',    SVC())
    ])
        
    param_dist = {
        'svc__kernel':       ['linear', 'poly', 'rbf'],                # discrete choices
        'svc__C':            uniform(1e-2, 1e2),                    # log-uniform between 0.01 and 100
        'svc__gamma':        uniform(1e-3, 1e1),                    # log-uniform between 0.001 and 10
        'svc__degree':       randint(1, 5),                         # integers 1–4 (only used if kernel='poly')
        'svc__coef0':        uniform(1e-3, 1),                      # shifts poly/sigmoid
    }

    model = SVC()
    return RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=3, random_state=42, error_score='raise')

def main():
    # getting the training and test data
    train_dirs = ["images/train/bmp", "images/train/btr", "images/train/cars", "images/train/grad", "images/train/howitzer", "images/train/tank"]
    test_dirs = ["test_images/bmp", "test_images/btr", "test_images/cars", "test_images/grad", "test_images/howitzer", "test_images/tank"]
    X_train, y_train = readData(train_dirs, class_part_index=2)
    X_test, y_test = readData(test_dirs, class_part_index=1)

    model = getModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print((y_pred == y_test).sum() / len(y_test))

main()