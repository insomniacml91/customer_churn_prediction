import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


df=pd.read_csv('churns.csv')

df = pd.get_dummies(df, columns=['Contract', 'InternetService'], drop_first=True)

binary_yes_no_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

df[binary_yes_no_cols] = df[binary_yes_no_cols].replace({
    'Male': 1, 'Female': 0,
    'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0
}).astype('int')
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype('int')
df = pd.get_dummies(df, columns=['PaymentMethod'], drop_first=True)

df = df.drop(columns=['TotalCharges', 'customerID'])

features = df.drop(columns=['Churn']).columns.tolist()

X = df[features]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Запуск RandomForest ---")
model = RandomForestClassifier(
    random_state=42, 
    class_weight='balanced', 
    max_depth=5, 
    n_estimators=100 
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Точность RandomForest: {accuracy*100:.1f}%')
print(classification_report(y_test, y_pred))


print("\n--- Запуск CatBoost ---")
cat_model = CatBoostClassifier(
    iterations=100, 
    depth=5, 
    class_weights=[1, 2.5], 
    verbose=0 
)
cat_model.fit(X_train, y_train)

cat_pred = cat_model.predict(X_test)

print(f'Точность CatBoost: {accuracy_score(y_test, cat_pred)*100:.1f}%')
print(classification_report(y_test, cat_pred))