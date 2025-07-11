import random
from typing import Literal
import numpy as np
import pandas as pd
import imageio.v3 as iio
from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from scipy.stats import uniform, randint
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
# from skimage import color
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def readData(dirs = [], class_part_index = 2, augment=False):
    img_paths = [Path(d) for d in dirs]
    paths = []
    for path in img_paths:
        paths += list(path.glob('*.jpeg'))
        paths += list(path.glob('*.jpg'))
        paths += list(path.glob('*.png'))
    random.shuffle(paths)

    def apply_transform(image) -> torch.Tensor:
        transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomVerticalFlip(p=0.3),
            T.RandomApply([T.RandomResizedCrop(size=(256, 256), scale=(0.6, 0.9))], p=0.5),
            T.RandomRotation(degrees=(-30, 30)),
            T.RandomApply([T.CenterCrop(100), T.Resize(size=(256, 256))], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.65)
        ])
        return transform(image)

    X = []
    y = []
    if augment == True:
        for path in paths:
            img_pre_torch = resize(iio.imread(path), (256, 256))
            X.append((img_pre_torch).flatten())
            y.append(path.parts[class_part_index])
            for i in range(3):
                img_torch: torch.Tensor = apply_transform(img_pre_torch)
                img_post_torch = np.transpose(img_torch.numpy(), (1, 2, 0))
                X.append((img_post_torch).flatten())
                y.append(path.parts[class_part_index])
    else:
        for path in paths:
            img_pre_torch = resize(iio.imread(path), (256, 256))
            X.append((img_pre_torch).flatten())
            y.append(path.parts[class_part_index])
    
    X = pd.DataFrame(X)
    y = pd.Series(y)
    return X, y

def getModel():
    class TunableAdaBoost(BaseEstimator, ClassifierMixin):
        def __init__(
            self,
            criterion: Literal['gini', 'entropy', 'log_loss']='entropy',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            n_estimators=50,
            learning_rate=1.0,
            random_state=None,
        ):
            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.class_weight = class_weight
            self.ccp_alpha = ccp_alpha
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.random_state = random_state

            self.model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                    max_features=self.max_features,
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_impurity_decrease=self.min_impurity_decrease,
                    class_weight=self.class_weight,
                    ccp_alpha=self.ccp_alpha,
                    random_state=self.random_state
                ),
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
        def fit(self, X, y):
            self.model.fit(X, y)
            self.is_fitted_ = True
            return self
        def predict(self, X):
            return self.model.predict(X)
        
    pipe = Pipeline([
        ('scaler', StandardScaler()),   # zero mean, unit variance
        ('pca', PCA(n_components=0.9)),
        ('svc', SVC())
    ])
    param_dist = {
        'svc__kernel':       ['linear', 'rbf'],                # discrete choices
        'svc__C':            uniform(1e-2, 0.1),                    # log-uniform between 0.01 and 100
        'svc__gamma':        uniform(1e-3, 1e-1),                    # log-uniform between 0.001 and 10
        'svc__degree':       randint(1, 5),                         # integers 1–4 (only used if kernel='poly')
        'svc__coef0':        uniform(1e-3, 1),                      # shifts poly/sigmoid
        # 'adaboost__n_estimators': uniform(2, 10),
        # 'adaboost__min_samples_leaf': randint(3, 10),
        # 'adaboost__min_impurity_decrease': uniform(0, 1),
        # 'adaboost__learning_rate': uniform(0.1, 1),
        # 'adaboost__max_depth': randint(50, 100),
        # 'adaboost__min_samples_split': randint(3, 15),
    }

    # return RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=3, random_state=42, error_score='raise')
    return RandomizedSearchCV(pipe, param_dist, n_iter=30, cv=StratifiedKFold(5), random_state=42, error_score='raise')

def main():
    # getting the training and test data
    train_dirs = [
        "images/train/bmp", "images/train/btr", "images/train/cars",
        "images/train/grad", "images/train/howitzer", "images/train/tank"
    ]
    test_dirs = ["test_images/bmp", "test_images/btr", "test_images/cars", "test_images/grad", "test_images/howitzer", "test_images/tank"]
    X_train, y_train = readData(train_dirs, class_part_index=2, augment=True)
    X_test, y_test = readData(test_dirs, class_part_index=1)

    model = getModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    print((y_pred == y_train).sum()/len(y_train))

    y_pred = model.predict(X_test)
    print((y_pred == y_test).sum()/len(y_test))

main()