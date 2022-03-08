import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model

warnings.filterwarnings("ignore")


def load_data(target, path):
    df = pd.read_csv(path)
    str_col = [col for col in df.columns if df[col].dtype == 'object']
    df_nontree = pd.get_dummies(df, columns=str_col, drop_first=False)

    X, y = df_nontree.drop(target, axis=1), df_nontree[target]

    X = X.apply(lambda x: x.fillna(x.mean()), axis=0)

    return X, y


def load_data_heartFailurePrediction():
    return load_data('target', 'Datasets/heartFailurePrediction.csv')


def load_data_HeartDiseaseUCI():
    return load_data('HeartDisease', 'Datasets/HeartDiseaseUCI.csv')


def train_model_heart(X, y):
    kf = StratifiedKFold(n_splits=5)
    log = make_pipeline(StandardScaler(), LogisticRegression())
    log_score = cross_validate(log, X=X, y=y, cv=kf, scoring=['accuracy', 'recall', 'precision', 'f1'])
    log_score = pd.DataFrame(log_score).mean()
    log_score = log_score[2:]
    return log_score.tolist()


def load_data_house():
    X, y = load_data('SalePrice', 'Datasets/HousePrices/train.csv')

    # ===========================================================================
    # select some features
    # ===========================================================================
    features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF',
                '1stFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces',
                'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', 'PoolArea', 'YrSold']

    X = X[features]
    return X, y


def nutrelize_inf(num):
    if np.isinf(num):
        return sys.float_info.max
    else:
        return num


def train_model_house(X, y):
    y = np.log1p(y)

    X = X.to_numpy()
    y = y.to_numpy().reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    rms = mean_squared_error(y_test, y_pred)

    print("The rms score is %.5f" % rms)
    return [rms]


def load_data_Cardiotocography():
    data = pd.read_excel('Datasets/Cardiotocography.xls', sheet_name=1, skiprows=1)
    data.drop(data.iloc[:, :10], inplace=True, axis=1)
    data.drop(data.iloc[:, 22:33], inplace=True, axis=1)
    data = data.drop(['Unnamed: 31', 'Unnamed: 44'], axis=1)
    data.isna().sum()
    data = data.dropna()
    data = data.drop_duplicates()
    X = data.drop(['CLASS'], axis=1)
    y = data['CLASS']
    return X, y


def train_model_Cardiotocography(X, y):
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # We will produce synthetic data to balance the classes
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # We will apply Standard Scaling to our data
    scalar = StandardScaler()
    standardized_X_train = pd.DataFrame(scalar.fit_transform(X_train), columns=X_train.columns)
    standardized_X_test = pd.DataFrame(scalar.transform(X_test), columns=X_test.columns)

    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=42)
    clf_gini.fit(standardized_X_train, y_train)

    y_pred_gini = clf_gini.predict(standardized_X_test)

    return [accuracy_score(y_test, y_pred_gini)]
