# ================================
# IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

sns.set(style="whitegrid")

# ================================
# LOAD DATASET
# ================================
df = pd.read_csv("rainfall in india 1901-2015.csv")

print(df.info())
print(df.describe())

# =========================================================
# OBJECTIVE 1: DATA CLEANING AND HANDLING MISSING VALUES
# =========================================================
print("\nMissing Values:\n", df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

high_rainfall = df[df['Annual'] > 3000]
sorted_df = df.sort_values(by='Annual', ascending=False)

print("\nHigh Rainfall Records:\n", high_rainfall.head())

# Scatter Plot → Year vs Rainfall
sns.scatterplot(x='Year', y='Annual', data=df)
plt.title("Year vs Annual Rainfall")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.show()

# Bar Plot → Top rainfall regions (FIXED)
df.groupby('Subdivision')['Annual'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 Subdivisions by Average Rainfall")
plt.xlabel("Subdivision")
plt.ylabel("Annual Rainfall (mm)")
plt.xticks(rotation=45)
plt.show()

# Monsoon Contribution Graph
df['Monsoon %'] = (df['Jun-Sep'] / df['Annual']) * 100

plt.hist(df['Monsoon %'], bins=30)
plt.title("Monsoon Contribution to Annual Rainfall (%)")
plt.xlabel("Percentage Contribution")
plt.ylabel("Frequency")
plt.show()

# Pie Chart → Seasonal Contribution
season_avg = df[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].mean()

plt.pie(season_avg, labels=season_avg.index, autopct='%1.1f%%')
plt.title("Seasonal Contribution to Total Rainfall")
plt.show()

# =========================================================
# OBJECTIVE 2: REGION-WISE COMPARISON USING BOXPLOT
# =========================================================
top_regions = df['Subdivision'].value_counts().head(5).index

sns.boxplot(data=df[df['Subdivision'].isin(top_regions)],
            x='Subdivision', y='Annual')
plt.xticks(rotation=45)
plt.title("Subdivision-wise Rainfall Comparison")
plt.xlabel("Subdivision")
plt.ylabel("Annual Rainfall (mm)")
plt.show()

# =========================================================
# OBJECTIVE 3: OUTLIER DETECTION AND FIXING
# =========================================================
plt.hist(df['Annual'], bins=30)
plt.title("Distribution of Annual Rainfall")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.show()

Q1 = df['Annual'].quantile(0.25)
Q3 = df['Annual'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_before = df[(df['Annual'] < lower) | (df['Annual'] > upper)]
print("Outliers before:", len(outliers_before))

df.loc[df['Annual'] < lower, 'Annual'] = lower
df.loc[df['Annual'] > upper, 'Annual'] = upper

outliers_after = df[(df['Annual'] < lower) | (df['Annual'] > upper)]
print("Outliers after:", len(outliers_after))

plt.hist(df['Annual'], bins=30)
plt.title("Rainfall after Outlier Capping")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.show()

# =========================================================
# OBJECTIVE 4: MACHINE LEARNING (PREDICT RAINFALL)
# =========================================================
X = df[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']]
y = df['Annual']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# =========================================================
# OBJECTIVE 5: CORRELATION ANALYSIS (HEATMAP)
# =========================================================
selected_cols = ['Annual', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
corr = df[selected_cols].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Seasonal Rainfall Components")
plt.show()

# =========================================================
# HYPOTHESIS TESTING (T-TEST)
# =========================================================
before = df[df['Year'] < 1980]['Annual']
after = df[df['Year'] >= 1980]['Annual']

t_stat, p_value = stats.ttest_ind(before, after)

print("\nT-Test:")
print("T-stat:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Significant change in rainfall over time")
else:
    print("No significant change in rainfall")

# =========================================================
# Z-TEST
# =========================================================
sample = df['Annual']
z = (sample.mean()) / (sample.std()/np.sqrt(len(sample)))

print("Z value:", z)
