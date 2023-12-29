import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
print(df.head()) # retrieving the first five rows of the dataframe
print(df.info()) # succinct summary
print(df.describe())

## E.D.A (Exploratory Data Analysis)
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)

sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)

# to plot pairwise relationships
sns.pairplot(df, kind='scatter', plot_kws={'alpha':0.3})

# combination of  a Scatter Plot and a Regression Line
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha':0.3})
plt.show()

## Split data into train and test sets