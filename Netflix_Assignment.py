# Netflix User Analytics Assignment
week 2 assignment 1
# Name: Kisa Fatema Haider Sayed
# Enrollment Number: 03404092025
# College: IGDTUW

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# PART A: DATASET UNDERSTANDING
# ============================================================

# Q1. Load the dataset and display the first five records
df = pd.read_csv("Dataset_2.csv")
print("Q1. First five records:")
print(df.head())
print()

# Q2. Number of rows and columns
print("Q2. Number of rows and columns:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print()

# Q3. All column names
print("Q3. Column names:")
print(df.columns.tolist())
print()

# Q4. Numerical and categorical features
numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical = df.select_dtypes(include=['object']).columns.tolist()
print("Q4. Numerical features:", numerical)
print("    Categorical features:", categorical)
print()

# Q5. Check for missing values
print("Q5. Missing values in each column:")
print(df.isnull().sum())
print()

# ============================================================
# PART B: EXPLORATORY DATA ANALYSIS
# ============================================================

# Q6. Average age of users
print("Q6. Average age of users:", round(df['Age'].mean(), 2))
print()

# Q7. Average watch hours per week
print("Q7. Average watch hours per week:", round(df['WatchHoursPerWeek'].mean(), 2))
print()

# Q8. Average monthly spending
print("Q8. Average monthly spending: ₹", round(df['MonthlySpend'].mean(), 2))
print()

# Q9. Count of users in each subscription category
print("Q9. Users per subscription category:")
print(df['SubscriptionType'].value_counts())
print()

# Q10. Percentage of users who renewed subscriptions
renewed = (df['SubscriptionRenewed'].value_counts(normalize=True) * 100).round(2)
print("Q10. Subscription renewal percentage:")
print(renewed)
print()

# ============================================================
# PART C: DATA PREPARATION
# ============================================================

# Q11. Convert categorical features to numerical
df_encoded = df.copy()
le = LabelEncoder()
for col in categorical:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("Q11. Data after encoding categorical features:")
print(df_encoded.head())
print()

# Q12. Define feature set X and target variable y for subscription renewal
X_class = df_encoded.drop(columns=['UserID', 'SubscriptionRenewed', 'MonthlySpend'])
y_class = df_encoded['SubscriptionRenewed']
print("Q12. Feature set X (for classification):")
print(X_class.columns.tolist())
print("Target variable y: SubscriptionRenewed")
print()

# Q13. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)
print("Q13. Train/Test Split:")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
print()

# ============================================================
# PART D: DECISION TREE CLASSIFICATION
# ============================================================

# Q14. Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Q14. Decision Tree model trained successfully.")
print()

# Q15. Evaluate model using accuracy
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Q15. Decision Tree Accuracy: {round(dt_accuracy * 100, 2)}%")
print()

# Q16. Confusion Matrix
print("Q16. Confusion Matrix for Decision Tree:")
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp.plot()
plt.title("Decision Tree - Confusion Matrix")
plt.savefig("confusion_matrix_dt.png")
plt.show()
print("Interpretation: The matrix shows True Positives, True Negatives,")
print("False Positives, and False Negatives for subscription renewal prediction.")
print()

# ============================================================
# PART E: K-NEAREST NEIGHBORS (KNN)
# ============================================================

# Q17. Train KNN classifier with K=5
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("Q17. KNN model (K=5) trained successfully.")
print()

# Q18. Compare KNN accuracy with Decision Tree
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("Q18. Model Accuracy Comparison:")
print(f"Decision Tree Accuracy: {round(dt_accuracy * 100, 2)}%")
print(f"KNN Accuracy (K=5):     {round(knn_accuracy * 100, 2)}%")
if dt_accuracy > knn_accuracy:
    print("Decision Tree performed better for this dataset.")
else:
    print("KNN performed better for this dataset.")
print()

# ============================================================
# PART F: LINEAR REGRESSION
# ============================================================

# Q19. Train Linear Regression to predict monthly spending
X_reg = df_encoded.drop(columns=['UserID', 'MonthlySpend', 'SubscriptionRenewed'])
y_reg = df_encoded['MonthlySpend']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_r, y_train_r)
print("Q19. Linear Regression model trained successfully to predict MonthlySpend.")
print()

# Q20. Predict monthly spending for a new user
new_user = pd.DataFrame({
    'Age': [30],
    'Gender': [1],              # 1 = Male (after encoding)
    'SubscriptionType': [2],    # 2 = Premium (after encoding)
    'WatchHoursPerWeek': [15],
    'DevicesUsed': [2],
    'FavoriteGenre': [0],       # 0 = Action (after encoding)
    'AdClicks': [10]
})

predicted_spend = lr_model.predict(new_user)
print(f"Q20. Predicted Monthly Spending for new user: ₹{round(predicted_spend[0], 2)}")
print("Interpretation: Based on the user's age, subscription type, watch hours,")
print("and other features, the model predicts their monthly spending.")
print()

# ============================================================
# BUSINESS REFLECTION QUESTIONS
# ============================================================

print("=" * 60)
print("BUSINESS REFLECTION QUESTIONS")
print("=" * 60)

print("""
1. Which factors appear to influence subscription renewal the most?
   - WatchHoursPerWeek, MonthlySpend, SubscriptionType, and AdClicks
     appear to be strong indicators of whether a user renews their
     subscription. Users who watch more and spend more are more
     likely to renew.

2. Why is subscription renewal a classification problem?
   - Because the output (SubscriptionRenewed) is a categorical variable
     with two classes: 'Yes' or 'No'. Classification algorithms predict
     which class a data point belongs to.

3. Why is monthly spending a regression problem?
   - Because MonthlySpend is a continuous numerical value (e.g., ₹353,
     ₹804). Regression predicts a continuous output, not a category.

4. Which algorithm performed better for renewal prediction?
   - Based on our results, the algorithm with higher accuracy performed
     better. Decision Trees are interpretable while KNN relies on
     distance-based similarity. Results may vary by dataset.

5. How could the platform use these predictions to improve customer retention?
   - Netflix can identify users likely NOT to renew and offer them
     personalized discounts, content recommendations, or reminders.
     This helps reduce churn and increase long-term revenue.
""")
