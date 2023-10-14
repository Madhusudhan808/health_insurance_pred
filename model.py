import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# creating the model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import KBinsDiscretizer
import pickle

data = pd.read_csv("insurance.csv")

data.head()

data.info()

data.describe()

data.hist(bins=50, figsize=(20,15))
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(data['sex'])
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(data['smoker'])
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(data['region'])
plt.show()

data.head()

data['sex'] = data['sex'].map({'female':0, 'male':1})
data['smoker'] = data['smoker'].map({'yes':1, 'no': 0})
data['region'] = data['region'].map({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4})

data.head()

features = data.select_dtypes(include=[np.number])
features.columns

# checking if any null values in  the data set
data.isnull().sum()

data.describe()

corrmat = data.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corrmat, annot=True, cmap="RdYlGn")

# so the data set is good lets train and test the data
x = data.drop(['charges'], axis=1)
y = data['charges']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

lr = LinearRegression()
lr.fit(x_train, y_train)
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

# Saving model to disk
pickle.dump(rf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(rf.predict(x_test))