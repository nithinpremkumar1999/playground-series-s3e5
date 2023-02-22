# Import Libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("../data/raw_data/train.csv") #import train dataset
test = pd.read_csv("../data/raw_data/test.csv") #import test dataset
print(train.head()) #display first 5 rows

#Rename column names, to make it more friendly for future
train.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)


def find_outliers(df):
    """
    Using traditional statistical method IQR to find outliers that lie below and above Q1-1.5*IQR and Q3+1.5IQR respectively,
    where Q1: 25th percentile, Q3: 75th percentile, and IQR: interquantile range
    IQR = Q3-Q1.
    :param df: pandas dataframe column
    :return: outliers : pandas dataframe
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3-q1
    outliers = df[((df < (q1-1.5*IQR)) | (df > (q3+1.5*IQR)))]
    return outliers


def find_lower_upper(df):
    """
    Assuming the column follows a normal distribution find upper and lower value which is within three standard
    deviations away from the mean.
    :param df: pandas dataframe column
    :return: lower: int, upper: int
    """
    upper = df.mean() + 3*df.std()
    lower = df.mean() - 3*df.std()
    return lower, upper


print("-----------------------")
#Id column
print("Id column")
print(train.Id.reset_index())

#Id and index are redundant, remove Id column
train.drop(["Id"], axis=1, inplace=True)
print("\n")
print(train.head())


print("-----------------------")
#fixed_acidity column
print("fixed_acidity column")

#check for NA values
print("NA values: ", train.fixed_acidity.isnull().sum())

#describe the column
print(train.fixed_acidity.describe())

#plot the column
plt.figure(1)
plt.hist(train.fixed_acidity)
plt.title("Histogram of fixed_acidity")
plt.savefig("../visualizations/data_preparation/fixed_acidity_hist.png")

#look for outliers since the max value is much greater than mean
outliers = find_outliers(train.fixed_acidity)
print("number of outliers: ", len(outliers))
print("min outlier value: ", outliers.min())
print("max outlier value:", outliers.max())

#the outliers are right skewed we can cap the values to 1 less than minimum of outliers
train.fixed_acidity = np.where(train.fixed_acidity >= outliers.min(), outliers.min()-1, train.fixed_acidity)

#describe the column after manipulation
print(train.fixed_acidity.describe())

print("-----------------------")
#voltile_acidity column
print("volatile_acidity column")

#check for NA values
print("NA values: ", train.volatile_acidity.isnull().sum())

#describe the column
print(train.volatile_acidity.describe())

#plot the column
plt.figure(2)
plt.hist(train.volatile_acidity)
plt.title("Histogram of volatile_acidity")
plt.savefig("../visualizations/data_preparation/volatile_acidity_hist.png")

outliers = find_outliers(train.volatile_acidity)
print("number of outliers: ", len(outliers))
print("min outlier value: ", outliers.min())
print("max outlier value:", outliers.max())

#the outliers are right skewed we can cap the values to 1 less than minimum of outliers
train.volatile_acidity = np.where(train.volatile_acidity >= outliers.min(), outliers.min()-1, train.volatile_acidity)

#describe the column after manipulation
print(train.volatile_acidity.describe())


print("-----------------------")
#citric_acidity column
print("citric_acid column")

#check for NA values
print("NA values: ", train.citric_acid.isnull().sum())

#describe the column
print(train.citric_acid.describe())

#plot the column
plt.figure(3)
plt.hist(train.citric_acid)
plt.title("Histogram of citric_acid")
plt.savefig("../visualizations/data_preparation/citric_acid_hist.png")

outliers = find_outliers(train.citric_acid)
print("number of outliers: ", len(outliers))
print("min outlier value: ", outliers.min())
print("max outlier value:", outliers.max())

print("-----------------------")
#residual_sugar column
print("residual_sugar column")

#check for NA values
print("NA values: ", train.residual_sugar.isnull().sum())

#describe the column
print(train.residual_sugar.describe())

#plot the column
plt.figure(4)
plt.hist(train.residual_sugar)
plt.title("Histogram of residual_sugar")
plt.savefig("../visualizations/data_preparation/residual_sugar_hist.png")

# refer reference 1.


print("-----------------------")
#chlorides column
print("chlorides column")

#check for NA values
print("NA values: ", train.chlorides.isnull().sum())

#describe the column
print(train.chlorides.describe())

#plot the column
plt.figure(5)
plt.hist(train.chlorides)
plt.title("Histogram of chlorides")
plt.savefig("../visualizations/data_preparation/chlorides_hist.png")

# chlorides concentration seems natural based on 2.


print("-----------------------")
#free_sulfur_dioxide column
print("free_sulfur_dioxide column")

#check for NA values
print("NA values: ", train.free_sulfur_dioxide.isnull().sum())

#describe the column
print(train.free_sulfur_dioxide.describe())

#plot the column
plt.figure(6)
plt.hist(train.free_sulfur_dioxide)
plt.title("Histogram of free_sulfur_dioxide")
plt.savefig("../visualizations/data_preparation/free_sulfur_dioxide_hist.png")

outliers = find_outliers(train.free_sulfur_dioxide)
print("number of outliers: ", len(outliers))
print("min outlier value: ", outliers.min())
print("max outlier value:", outliers.max())

#drop the outliers since it is very less
train.drop(outliers.index, inplace=True)


print("-----------------------")
#total_sulfur_dioxide column
print("total_sulfur_dioxide column")

#check for NA values
print("NA values: ", train.total_sulfur_dioxide.isnull().sum())

#describe the column
print(train.total_sulfur_dioxide.describe())

#plot the column
plt.figure(7)
plt.hist(train.total_sulfur_dioxide)
plt.title("Histogram of total_sulfur_dioxide")
plt.savefig("../visualizations/data_preparation/total_sulfur_dioxide_hist.png")

outliers = find_outliers(train.total_sulfur_dioxide)
print("number of outliers: ", len(outliers))
print("min outlier value: ", outliers.min())
print("max outlier value:", outliers.max())

# total_sulfur_dioxide concentration seems natural based on 3.


print("-----------------------")
#density column
print("density column")

#check for NA values
print("NA values: ", train.density.isnull().sum())

#describe the column
print(train.density.describe())

#plot the column
plt.figure(8)
plt.hist(train.density)
plt.title("Histogram of density")
plt.savefig("../visualizations/data_preparation/density_hist.png")

outliers = find_outliers(train.density)
print("number of outliers: ", len(outliers))
print("min outlier value: ", outliers.min())
print("max outlier value:", outliers.max())

# density value is very close to mean value, most of the values are centered at the center


print("-----------------------")
#pH column
print("pH column")

#check for NA values
print("NA values: ", train.pH.isnull().sum())

#describe the column
print(train.pH.describe())

#plot the column
plt.figure(9)
plt.hist(train.pH)
plt.title("Histogram of pH")
plt.savefig("../visualizations/data_preparation/pH_hist.png")

outliers = find_outliers(train.pH)
print("number of outliers: ", len(outliers))
print("min outlier value: ", outliers.min())
print("max outlier value:", outliers.max())

# pH values are within regulated values, refer 4


print("-----------------------")
#sulphates column
print("sulphates column")

#check for NA values
print("NA values: ", train.sulphates.isnull().sum())

#describe the column
print(train.sulphates.describe())

#plot the column
plt.figure(10)
plt.hist(train.sulphates)
plt.title("Histogram of sulphates")
plt.savefig("../visualizations/data_preparation/sulphates_hist.png")

#remove values greater than 1.4
index = train[train.sulphates >= 1.4].index
train.drop(index, inplace=True)

outliers = find_outliers(train.sulphates)
print("number of outliers: ", len(outliers))
print("min outlier value: ", outliers.min())
print("max outlier value:", outliers.max())

#cap values at 95th percentile
_, upper = find_lower_upper(train.sulphates)
train.sulphates = np.where(train.sulphates > upper, upper, train.sulphates)


print("-----------------------")
#alcohol column
print("alcohol column")

#check for NA values
print("NA values: ", train.alcohol.isnull().sum())

#describe the column
print(train.alcohol.describe())

#plot the column
plt.figure(11)
plt.hist(train.alcohol)
plt.title("Histogram of alcohol")
plt.savefig("../visualizations/data_preparation/alcohol_hist.png")

# alcohol is within range of average alcohol concentration in wine


print("-----------------------")
#quality column
print("quality column")

#check for NA values
print("NA values: ", train.quality.isnull().sum())


#rename test dataset to make it uniform
test.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)


#export train and test as csv
train.to_csv("../data/clean_data/train.csv")
test.to_csv("../data/clean_data/test.csv", index=False)






