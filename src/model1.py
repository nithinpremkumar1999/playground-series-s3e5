from target_mapping import *
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
import joblib
import pandas as pd
pd.set_option('display.max_columns', None)

train = pd.read_csv("../data/feature_engineered_data/train.csv")
test = pd.read_csv("../data/feature_engineered_data/test.csv")

X, y = train.loc[:, train.columns!="quality"], train["quality"]
#scale target variable
y = target_to_encoding(y)
X_test = test.loc[:, test.columns != "Id"]

transformer = QuantileTransformer(output_distribution="normal")
X = transformer.fit_transform(X)
X_test = transformer.transform(X_test)


pipeline = Pipeline(steps=[['smote', SMOTE(random_state=15)],
                    ['classifier', LogisticRegression(random_state=15, multi_class="multinomial", solver="newton-cg")]])


stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)


param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           scoring='roc_auc_ovr',
                           cv=stratified_kfold,
                           n_jobs=-1)


grid_search.fit(X, y)

submission = grid_search.predict(X_test)
submission = pd.DataFrame(submission, columns=["quality"])
submission["quality"] = encoding_to_target(submission["quality"])
submission["Id"] = test["Id"]

print("Cross-Validated ROC_AUC score:", grid_search.best_score_)
submission.to_csv("../data/submissions/submission_model1.csv", index=False)
joblib.dump(grid_search.best_estimator_, "../models/model1.pkl")