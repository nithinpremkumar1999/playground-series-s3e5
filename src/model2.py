from target_mapping import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pandas as pd
pd.set_option('display.max_columns', None)

train = pd.read_csv("../data/feature_engineered_data/train.csv")
test = pd.read_csv("../data/feature_engineered_data/test.csv")

X, y = train.loc[:, train.columns!="quality"], train["quality"]
#scale target variable
y = target_to_encoding(y)

pipeline = Pipeline(steps=[['smote', SMOTE(random_state=10)],
                    ['classifier', KNeighborsClassifier()]])


stratified_kfold = StratifiedKFold(n_splits=12, shuffle=True, random_state=14)


param_grid = {'classifier__n_neighbors': list(range(1, 30))}


grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           scoring='roc_auc_ovr',
                           cv=stratified_kfold,
                           n_jobs=-1)


grid_search.fit(X, y)


submission = grid_search.predict(test.loc[:, test.columns != "Id"])
submission = pd.DataFrame(submission, columns=["quality"])
submission["quality"] = encoding_to_target(submission["quality"])
submission["Id"] = test["Id"]


print("Cross-Validated ROC_AUC score:", grid_search.best_score_)
#submission.to_csv("../data/submissions/submission_model2.csv", index=False)
joblib.dump(grid_search.best_estimator_, "../models/model2.pkl")
