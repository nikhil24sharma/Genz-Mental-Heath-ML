#   main goal is to predict the mental health score 
#  libraries importing

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# data loading

data  =  pd.read_csv(r"D:\Data Science projects\GenZ social Media  usage\genz_social_media_usage_1M.csv")
# print(data.shape)
# print(data.isnull().sum())

# data exportation to excel


# with pd.ExcelWriter(r"D:\Data Science projects\GenZ social Media  usage\genz_social_media_usage_1M.xlsx") as writer:
#     data.head(50).to_excel(writer, sheet_name="Head", index=False)
    
#     # tail
#     data.tail(50).to_excel(writer, sheet_name="Tail", index=False)
    
#     # describe
#     data.describe().to_excel(writer, sheet_name="Describe")

# data visulization
# numerical data distribution

# sns.histplot(data['daily_usage_hours'] , kde  = True)
# plt.show()

# # categorical data distribution
# sns.countplot(x='purpose', data=data)
# plt.show()


# sns.scatterplot(x='daily_usage_hours', y='mental_health_score', data=data)
# plt.show()

# print(data.corr(numeric_only=True)['mental_health_score'])

# plt.scatter(data["daily_usage_hours"], data["mental_health_score"])
# plt.xlabel("daily_usage_hours")
# plt.ylabel("mental_health_score")
# plt.show()



# # print(data.head(10))
# print(data.corr(numeric_only=True)['mental_health_score'])


# encodeing the categorical data

# ordinal encoding for addiction level

encoder = OrdinalEncoder(
    categories = [['Low', 'Medium', 'High']]
    # here it gives low = 0, medium = 1 and high is 2


)
data['addiction_level_encoded'] = encoder.fit_transform(data[['addiction_level']])
# print(data.head(10))


ohe = pd.get_dummies(
    data[['gender', 'country', 'primary_platform', 'purpose']],
    drop_first = True

)
numeric_col = data[['age','daily_usage_hours','num_platforms_used','avg_session_minutes','night_usage', 'screen_time_before_sleep', 'addiction_level_encoded']]

# combine the encoded categorical data with the numeric data

X = pd.concat([numeric_col,ohe] , axis = 1)
y = data['mental_health_score']

# print(X.shape)
# print(X.columns.tolist())

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# checking the performance of the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, lr_pred)
r2 = r2_score(y_test, lr_pred)
print("Mean Squared Error lr:", mse)
print("R-squared lr:", r2)

# random forest regressor has to make 100 trees for 1 lakh rows so we use random some value
# taking sample from already encoded data

X_sample = X.sample(n=10000, random_state=42)
y_sample = y.loc[X_sample.index]

# train test split

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42
)

# Random Forest Model

rf = RandomForestRegressor(
    n_estimators=10,
    random_state=42
)

# training

rf.fit(X_train1, y_train1)

# prediction

rf_pred = rf.predict(X_test1)

# evaluation

mse_rf = mean_squared_error(y_test1, rf_pred)
r2_rf = r2_score(y_test1, rf_pred)

print("Random Forest Mean Squared Error:", mse_rf)
print("Random Forest R-squared:", r2_rf)


