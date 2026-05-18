# ================================
# CUSTOMER CHURN PREDICTION PROJECT
# ================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import seaborn as sns 

# STEP 2: LOAD DATASET
# Replace the file name with your CSV file name

df = pd.read_csv("Telco_Customer_Churn.csv")

# STEP 3: DISPLAY FIRST 5 ROWS
print("FIRST 5 ROWS")
print(df.head())

# STEP 4: CHECK MISSING VALUES
print("\nMISSING VALUES")
print(df.isnull().sum())

# STEP 5: REMOVE UNNECESSARY COLUMN
# customerID is not useful for prediction

df.drop(["CustomerID","Churn Label", "Churn Score", "Churn Reason"], axis=1, inplace=True)

# STEP 6: HANDLE TOTALCHARGES COLUMN
# Convert object to numeric

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# Fill missing values with mean

df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].mean())

# STEP 7: CONVERT TEXT DATA INTO NUMBERS

encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])

# STEP 8: DEFINE FEATURES AND TARGET

X = df.drop("Churn Value", axis=1)
y = df["Churn Value"]

# Feature Scaling

scaler = StandardScaler()
X = scaler.fit_transform(X)

# STEP 9: SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# STEP 10: CREATE MODEL

model = LogisticRegression(max_iter=1000)

# STEP 11: TRAIN MODEL

model.fit(X_train, y_train)

# STEP 12: PREDICT OUTPUT

y_pred = model.predict(X_test)

# STEP 13: CHECK ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print(accuracy)

# STEP 14: CONFUSION MATRIX

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# STEP 15: CLASSIFICATION REPORT

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

# STEP 16: SAMPLE PREDICTIONS

print("\nSAMPLE PREDICTIONS")
print(y_pred[:10])

# STEP:17 SIMPLE VISUALIZATION

churn_counts = df["Churn Value"].value_counts()
plt.bar(["Stayed","Left"],churn_counts)
plt.title("Customer Churn Count")
plt.xlabel("Customer Status")
plt.ylabel("Count")
plt.show()