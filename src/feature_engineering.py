import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA


#we only need independent variable
train = pd.read_csv("../data/clean_data/train.csv", usecols=range(1, 13))
test = pd.read_csv("../data/clean_data/test.csv")


#drop variables that are not useful for predictions
train.drop(["fixed_acidity", "free_sulfur_dioxide", "pH", "chlorides", "residual_sugar"], axis=1, inplace=True)
test.drop(["fixed_acidity", "free_sulfur_dioxide", "pH", "chlorides", "residual_sugar"], axis=1, inplace=True)


#perform quantile transformer on variables non-normal variables
#Quantile Transformation

#total_sulfur_dioxide
quantile_total_sulfur_dioxide =  QuantileTransformer(output_distribution="normal").fit(train["total_sulfur_dioxide"]
                                                                                       .values.reshape(-1, 1))

    #train
train["trans_total_sulfur_dioxide"] = quantile_total_sulfur_dioxide.transform(train["total_sulfur_dioxide"]
                                                                              .values.reshape(-1, 1))
train.drop(["total_sulfur_dioxide"], axis=1, inplace=True)
    #test
test["trans_total_sulfur_dioxide"] = quantile_total_sulfur_dioxide.transform(test["total_sulfur_dioxide"]
                                                                              .values.reshape(-1, 1))
test.drop(["total_sulfur_dioxide"], axis=1, inplace=True)

#alcohol
quantile_alcohol = QuantileTransformer(output_distribution="normal").fit(train["alcohol"].values.reshape(-1, 1))

    #train
train["trans_alcohol"] = quantile_alcohol.transform(train["alcohol"].values.reshape(-1, 1))
train.drop(["alcohol"], axis=1, inplace=True)
    #test
test["trans_alcohol"] = quantile_alcohol.transform(test["alcohol"].values.reshape(-1, 1))
test.drop(["alcohol"], axis=1, inplace=True)

#citric_acid
quantile_citric_acid = QuantileTransformer(output_distribution="normal").fit(train["citric_acid"].values.reshape(-1, 1))

    #train
train["trans_citric_acid"] = quantile_citric_acid.transform(train["citric_acid"].values.reshape(-1, 1))
train.drop(["citric_acid"], axis=1, inplace=True)
    #test
test["trans_citric_acid"] = quantile_citric_acid.transform(test["citric_acid"].values.reshape(-1, 1))
test.drop(["citric_acid"], axis=1, inplace=True)


#combine features
#Principal Component Analysis

#alcohol&sulphates
pca_1 = PCA(n_components=2).fit(train[["trans_alcohol", "sulphates"]])

    #train
train[["pca1", "pca2"]] = pca_1.transform(train[["trans_alcohol", "sulphates"]])
train.drop(["trans_alcohol", "sulphates"], axis=1, inplace=True)
    #test
test[["pca1", "pca2"]] = pca_1.transform(test[["trans_alcohol", "sulphates"]])
test.drop(["trans_alcohol", "sulphates"], axis=1, inplace=True)


#total_sulfur_dioxide, volatile_acidity, density
pca_2 = PCA(n_components=3).fit(train[["volatile_acidity", "trans_total_sulfur_dioxide", "density"]])

    #train
train[["pca3", "pca4", "pca5"]] = pca_2.transform(train[["volatile_acidity", "trans_total_sulfur_dioxide", "density"]])
train.drop(["volatile_acidity", "trans_total_sulfur_dioxide", "density"], axis=1, inplace=True)
    #test
test[["pca3", "pca4", "pca5"]] = pca_2.transform(test[["volatile_acidity", "trans_total_sulfur_dioxide", "density"]])
test.drop(["volatile_acidity", "trans_total_sulfur_dioxide", "density"], axis=1, inplace=True)


#export train and test dataset
train.to_csv("../data/feature_engineered_data/train.csv", index=False)
test.to_csv("../data/feature_engineered_data/test.csv", index=False)




