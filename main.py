import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

### Problem Statement
# This dataset is having data of customers who buys clothes online.
# The store offers in-store style and clothing advice sessions. Customers come in to the store,
# have sessions/meetings with a personal stylist, then they can go home and order either on a
# mobile app or website for the clothes they want.
# The company is trying to decide whether to focus their efforts
# on their mobile app experience or their website.

### Dataset

# Loading the data
df = pd.read_csv("ecommerce.csv")

# Viewing the data
# print(df.head()) # retrieving the first five rows of the dataframe
# print(df.info()) # succinct summary
# print(df.describe())

## E.D.A (Exploratory Data Analysis)
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)

sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)

# to plot pairwise relationships
sns.pairplot(df, kind='scatter', plot_kws={'alpha':0.3})

# combination of a Scatter Plot and a Regression Line
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha':0.3})
#plt.show()

## We split data into training and testing sets

# feature variables
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# print(X_test)
# print(y_test)

## Creating a linear regression model
model = LinearRegression()

## Then we fit the model with the training data
model.fit(X_train, y_train)

# Display the coefficients
coeff = pd.DataFrame(model.coef_, X.columns, columns=['Coefficients'])
#print(coeff)

## Making predictions on the test data set
predictions = model.predict(X_test)

# graph to see if the model fits the data
sns.scatterplot(x=predictions, y=y_test)
plt.xlim(200)
plt.xlabel("Predictions")
plt.title("Evaluation of our Linear Regression Model")
#plt.show()

## Evaluation of the model with metrics
print("Mean Absolute Error : ", mean_absolute_error(y_test, predictions))
print("Mean Squared Error : ", mean_squared_error(y_test, predictions))
print("Root Mean Squared Error : ", math.sqrt(mean_squared_error(y_test, predictions)))