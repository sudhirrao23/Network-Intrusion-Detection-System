import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('Monday-WorkingHours.pcap_ISCX.csv')
except FileNotFoundError:
    print("Error: 'Monday-WorkingHours.pcap_ISCX.csv' not found.")
    print("Please download the CIC-IDS-2017 dataset and place the file in the correct directory.")
    exit()

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
df.drop('Label', axis=1, inplace=True)

X = df.drop('label', axis=1)
y = df['label']

object_columns = X.select_dtypes(include=['object']).columns
X = X.drop(columns=object_columns)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("Training the NIDS model...")
model.fit(X_train, y_train)
print("Training complete.")

print("\nEvaluating the model...")
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack'], labels=[0, 1]))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
plt.title('Confusion Matrix for NIDS (CIC-IDS-2017)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()





