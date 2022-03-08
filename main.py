from source.outliers_detector_utils import *
from source.outliers_detector import OutlierGridSearcher

if __name__ == '__main__':
    #*** HousePrices ***#
    ogsHousePrices = OutlierGridSearcher(train_model_house, higher_is_better=False)
    X_train, y_train = load_data_house()
    X_filtered, y_filtered = ogsHousePrices.fit(X_train, y_train, top_nums_to_remove=[0, 10, 20, 30])

    #*** HeartDiseaseUCI ***#
    ogsHeartDiseaseUCI = OutlierGridSearcher(train_func=train_model_heart, scoring_weights=[0.4, 0.0, 0.1, 0.5])
    X_train, y_train = load_data_HeartDiseaseUCI()
    X_filtered, y_filtered = ogsHeartDiseaseUCI.fit(X_train, y_train)

    #*** heartFailurePrediction ***#
    ogsheartFailurePrediction = OutlierGridSearcher(train_func=train_model_heart)
    X_train, y_train = load_data_heartFailurePrediction()
    X_filtered, y_filtered = ogsheartFailurePrediction.fit(X_train, y_train, top_nums_to_remove=[0, 5, 10])

    #*** Cardiotocography ***#
    ogsCardiotocography = OutlierGridSearcher(train_func=train_model_Cardiotocography)
    X_train, y_train = load_data_Cardiotocography()
    X_filtered, y_filtered = ogsCardiotocography.fit(X_train, y_train, top_nums_to_remove=[0, 10, 30, 50, 100, 150, 200])

