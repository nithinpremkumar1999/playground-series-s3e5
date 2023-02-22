from imblearn.ensemble import BalancedBaggingClassifier
from target_mapping import *
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import joblib


train = pd.read_csv("../data/feature_engineered_data/train.csv")
test = pd.read_csv("../data/feature_engineered_data/test.csv")


X, y = train.loc[:, train.columns != "quality"], train["quality"]
y = target_to_encoding(y)
X_test = test.loc[:, test.columns != "Id"]

transformer = QuantileTransformer(output_distribution="normal")
X = transformer.fit_transform(X)
X_test = transformer.transform(X_test)


model = BalancedBaggingClassifier(estimator=LogisticRegression(random_state=15,
                                                               multi_class="multinomial",
                                                               solver="newton-cg"),
                                  n_estimators=20,
                                  replacement=False,
                                  sampling_strategy='auto',
                                  random_state=19)


stratified_kfold = StratifiedKFold(n_splits=12, shuffle=True, random_state=14)

cross_val = cross_validate(model, X, y, scoring="roc_auc_ovr", cv=stratified_kfold)

print("Cross-Validated ROC_AUC score:", cross_val)


model.fit(X, y)
submission = model.predict(X_test)
submission = pd.DataFrame(submission, columns=["quality"])
submission["quality"] = encoding_to_target(submission["quality"])
submission["Id"] = test["Id"]


submission.to_csv("../data/submissions/submission_model4.csv", index=False)
joblib.dump(model, "../models/model4.pkl")
