# ==============================================================================
# CELL 1: Setup and Data Generation
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set a seed for reproducibility
np.random.seed(42)
N = 500  # Total data points

# Define normal operating parameters
MEAN_TEMP = 50
MEAN_VIB = 1.2
MEAN_PRES = 15

# Generate primary features (normally distributed sensor data)
temp = np.random.normal(loc=MEAN_TEMP, scale=5, size=N)
vibration = np.random.normal(loc=MEAN_VIB, scale=0.3, size=N)
pressure = np.random.normal(loc=MEAN_PRES, scale=2, size=N)
hours = np.random.randint(500, 10000, size=N) # Operating hours

# Target: 90% Normal (0), 10% Failure (1)
failure_flag = np.random.choice([0, 1], size=N, p=[0.90, 0.10])

# Introduce correlation: Artificially increase/decrease features for failed instances
for i in np.where(failure_flag == 1)[0]:
    temp[i] += np.random.randint(10, 20)      # High temp before failure
    vibration[i] += np.random.uniform(1.0, 2.5) # High vibration
    pressure[i] -= np.random.uniform(5, 10)   # Low pressure drop (e.g., leak)

# Create the final DataFrame
data = pd.DataFrame({
    'Temperature': temp,
    'Vibration_Level': vibration,
    'Pressure_Drop': pressure,
    'Operating_Hours': hours,
    'Failure_Flag': failure_flag
})

print("--- Sample Historical Data (First 5 Rows) ---")
print(data.head())
print("\nFailure Count (Target Variable):")
print(data['Failure_Flag'].value_counts())


# ==============================================================================
# CELL 2: Exploratory Data Analysis (EDA)
# ==============================================================================

# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Failure_Flag', data=data)
plt.title('Distribution of Failure Flag (0=Normal, 1=Failure)')
plt.show()

# Visualize a key feature (Temperature) based on the target class
plt.figure(figsize=(8, 5))
sns.boxplot(x='Failure_Flag', y='Temperature', data=data)
plt.title('Temperature Distribution by Failure Status')
plt.show()

# Correlate features with the Failure Flag
correlation_matrix = data.corr()
print("\n--- Correlation Matrix (Focus on 'Failure_Flag' row) ---")
print(correlation_matrix['Failure_Flag'].sort_values(ascending=False))


# ==============================================================================
# CELL 3: Data Preprocessing and Splitting
# ==============================================================================

# Define features (X) and target (y)
X = data.drop('Failure_Flag', axis=1) # Features are all columns EXCEPT 'Failure_Flag'
y = data['Failure_Flag']              # Target variable

# Split data into training and testing sets (70% train, 30% test)
# 'stratify=y' ensures both train and test sets have the same proportion of failures
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Total instances: {N}")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


# ==============================================================================
# CELL 4: Model Training (Random Forest Classifier)
# ==============================================================================

# Initialize the classifier (Random Forest is good for classification tasks)
# n_estimators is the number of trees in the forest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("Starting Model Training...")

# Train the model using the training data
model.fit(X_train, y_train)

print("Model Training Complete.")


# ==============================================================================
# CELL 5: Prediction and Model Assessment
# ==============================================================================

# Make predictions on the unseen test data
y_pred = model.predict(X_test)

# --- Assessment ---
print("\n--- Model Assessment (Classification Report) ---")
# Precision, Recall, F1-Score are key metrics, especially for the '1' class (Failure)
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Failure (1)']))

print("\n--- Confusion Matrix ---")
# [True Negatives, False Positives]
# [False Negatives, True Positives]
print(confusion_matrix(y_test, y_pred))

