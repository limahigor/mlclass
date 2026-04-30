from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE


def build_preprocessors(numeric_cols, categorical_cols):
    preprocessor_linear = ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ]),
                categorical_cols,
            ),
        ]
    )

    preprocessor_tree = ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                ]),
                numeric_cols,
            ),
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ]),
                categorical_cols,
            ),
        ]
    )
    return preprocessor_linear, preprocessor_tree


def build_models(numeric_cols, categorical_cols):
    preprocessor_linear, preprocessor_tree = build_preprocessors(numeric_cols, categorical_cols)
    return {
        'logistic_regression': Pipeline([
            ('prep', preprocessor_linear),
            ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        'lda': Pipeline([
            ('prep', preprocessor_linear),
            ('model', LinearDiscriminantAnalysis()),
        ]),
        'knn_11': Pipeline([
            ('prep', preprocessor_linear),
            ('model', KNeighborsClassifier(n_neighbors=11)),
        ]),
        'svm_rbf': Pipeline([
            ('prep', preprocessor_linear),
            ('model', SVC(kernel='rbf', C=3.0, gamma='scale', random_state=RANDOM_STATE)),
        ]),
        'decision_tree': Pipeline([
            ('prep', preprocessor_tree),
            ('model', DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE)),
        ]),
        'random_forest': Pipeline([
            ('prep', preprocessor_tree),
            ('model', RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }
