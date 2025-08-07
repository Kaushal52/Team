import pandas as pd

# Path to the CSV file
file_path = "/Users/kaushalkhatu/Desktop/TEAM/DT.CSV"

# Read the CSV file
df = pd.read_csv(file_path)

# Show basic info and first few rows
print("Shape:", df.shape)
print(df.head())

import pandas as pd

# Path to the original CSV file
file_path = "/Users/kaushalkhatu/Desktop/TEAM/DT.CSV"

# Read the original CSV file
df = pd.read_csv(file_path)

# Remove players with 0 points
df_cleaned = df[df['POINTS'] > 0]

# Convert the 'CREDITS' column from float to int
df_cleaned['CREDITS'] = pd.to_numeric(df_cleaned['CREDITS'], errors='coerce')

# Drop rows where 'CREDITS' could not be converted to a valid number (NaN values)
df_cleaned = df_cleaned.dropna(subset=['CREDITS'])

# Convert the 'CREDITS' column to int
df_cleaned['CREDITS'] = df_cleaned['CREDITS'].astype(int)

# Path to save the cleaned CSV file
cleaned_file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv(cleaned_file_path, index=False)

# Confirm the cleaned data has been saved
print(f"Cleaned data saved to {cleaned_file_path}")

import pandas as pd

# Path to the CSV file
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"

# Read the CSV file
df = pd.read_csv(file_path)

# Get a summary of the dataset
print("Summary of the dataset:")
print(df.info())

# Get unique values for each column
print("\nUnique values in each column:")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique())

# Get basic statistics for numerical columns (like POINTS, CREDITS)
print("\nBasic Statistics for Numerical Columns:")
print(df.describe())

# Get the count of unique values in each column
print("\nCount of unique values in each column:")
print(df.nunique())

import pandas as pd

# Path to the cleaned CSV file
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"

# Read the CSV file
df = pd.read_csv(file_path)

# Display the first few rows (head) of the DataFrame
print("First few rows of the dataset:")
print(df.head())

# Display the column names
print("\nColumn names in the dataset:")
print(df.columns)

import pandas as pd

# Load the file
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print column names with quotes to inspect actual names
for col in df.columns:
    print(f"'{col}'")

import pandas as pd

# Load the file
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Show all column names
print("All column names:")
print(df.columns.tolist())

# Check unique values in 'TEAM' and 'Role' columns
print("\nContent of the 'TEAM' column:")
print(df['TEAM'].unique())

# Check unique values in 'ROLE' column (case-sensitive issue fix)
print("\nContent of the 'ROLE' column:")
print(df['ROLE'].unique())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the cleaned CSV file
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"

# Read the CSV file
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# 1. Points vs. Credits with Player Names
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='CREDITS', y='POINTS', hue='TEAM', palette='Set2')

# Annotating players with their names
for i in range(len(df)):
    plt.text(df['CREDITS'][i] + 0.05, df['POINTS'][i], df['PLAYERS'][i], 
             horizontalalignment='left', size='small', color='black', weight='semibold')

plt.title('Points vs Credits (with Player Names)')
plt.xlabel('Credits')
plt.ylabel('Points')
plt.legend(title='Team')
plt.show()

# 2. Points vs. Team
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='TEAM', y='POINTS', palette='Set2')
plt.title('Points Distribution by Team')
plt.xlabel('Team')
plt.ylabel('Points')
plt.show()

# 3. Points vs. Role
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='ROLE', y='POINTS', palette='Set1')  # Fixed 'Role' to 'ROLE'
plt.title('Points Distribution by Role')
plt.xlabel('Role')
plt.ylabel('Points')
plt.show()

# 4. Points vs. Player (if the number of players is manageable)
if len(df) <= 50:  # Only plot if there are fewer than or equal to 50 players
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='PLAYERS', y='POINTS', palette='viridis')
    plt.title('Points by Player')
    plt.xlabel('Player')
    plt.ylabel('Points')
    plt.xticks(rotation=90)
    plt.show()
else:
    print("Too many players to display in a bar plot.")


import pandas as pd

# Load your cleaned dataset
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"
df = pd.read_csv(file_path)

# Display basic statistics for the POINTS column
points_stats = df['POINTS'].describe()
print("📈 Points Statistics:\n")
print(points_stats)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"
df = pd.read_csv(file_path)

# Plot histogram and KDE for POINTS
plt.figure(figsize=(10, 6))
sns.histplot(df['POINTS'], kde=True, color='skyblue', bins=10, edgecolor='black')

plt.title('Distribution of Player Points')
plt.xlabel('Points')
plt.ylabel('Number of Players')
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your cleaned dataset
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"
df = pd.read_csv(file_path)

# Compute correlation matrix (only for numeric columns)
correlation_matrix = df[['CREDITS', 'POINTS']].corr()

# Print correlation matrix
print("Correlation Matrix:\n")
print(correlation_matrix)

# Optional: visualize as heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Credits and Points')
plt.show()


import pandas as pd

# Load the cleaned dataset
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"
df = pd.read_csv(file_path)

# Calculate the average (mean) of points
average_points = df['POINTS'].mean()

# Display the result
print(f"Average Points: {average_points:.2f}")

import pandas as pd
import numpy as np

# Load the cleaned dataset
df = pd.read_csv("/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV")

# Step 1: Calculate average points
avg_points = df['POINTS'].mean()

# Step 2: Filter players with more than average points
df_filtered = df[df['POINTS'] > avg_points]

# Step 3: Select 2 Wicketkeepers (WK)
wk_players = df_filtered[df_filtered['ROLE'] == 'WK'].head(2)

# Step 4: Select 3 Batsmen (BT)
bt_players = df_filtered[df_filtered['ROLE'] == 'BT'].head(3)

# Step 5: Select 4 All-rounders (AR) or 3 if 3 bowlers are selected
ar_players = df_filtered[df_filtered['ROLE'] == 'AR'].head(4)

# Step 6: Select 2 Bowlers (BL)
bl_players = df_filtered[df_filtered['ROLE'] == 'BL'].head(2)

# Step 7: Ensure more players from RR than DC
rr_players = df_filtered[df_filtered['TEAM'] == 'RR']
dc_players = df_filtered[df_filtered['TEAM'] == 'DC']

rr_count = min(len(rr_players), 6)
dc_count = min(len(dc_players), 5)

# Adjust counts to make sure RR has 1-2 more than DC
if rr_count <= dc_count:
    rr_count = min(dc_count + 1, len(rr_players))
    dc_count = rr_count - 1

rr_players_selected = rr_players.head(rr_count)
dc_players_selected = dc_players.head(dc_count)

# Step 8: Combine all selected players
selected_players = pd.concat([
    wk_players, bt_players, ar_players, bl_players, 
    rr_players_selected, dc_players_selected
])

# Step 9: Remove duplicates
selected_players = selected_players.drop_duplicates(subset='PLAYERS')

# Step 10: Ensure team has max 11 players
if len(selected_players) > 11:
    selected_players = selected_players.sort_values(by='POINTS', ascending=False).head(11)

# Step 11: Add outlier with lowest credits if under 11
if len(selected_players) < 11:
    remaining = df_filtered[~df_filtered['PLAYERS'].isin(selected_players['PLAYERS'])]
    outlier_player = remaining.sort_values(by='CREDITS').iloc[0]
    selected_players = pd.concat([selected_players, pd.DataFrame([outlier_player])])

# Step 12: Sort by ROLE order
role_order = {'WK': 1, 'BT': 2, 'AR': 3, 'BL': 4}
selected_players['ROLE_ORDER'] = selected_players['ROLE'].map(role_order)
selected_players = selected_players.sort_values(by='ROLE_ORDER').drop(columns=['ROLE_ORDER'])

# Step 13: Display and save
print("Selected Dream Team Players:")
print(selected_players[['PLAYERS', 'TEAM', 'ROLE']])

selected_players[['PLAYERS', 'TEAM', 'ROLE']].to_csv("/Users/kaushalkhatu/Desktop/TEAM/Dream_Team.csv", index=False)




import pandas as pd

# Load the DataFrame (replace with your actual file path)
file_path = "/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV"
df = pd.read_csv(file_path)

# Display the columns of the DataFrame
print("Columns in the DataFrame:")
print(df.columns)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load data
# df = pd.read_csv("your_file.csv") # if not already loaded

# Create binary target: 1 if above average points, else 0
avg_points = df['POINTS'].mean()
df['HIGH_SCORER'] = (df['POINTS'] > avg_points).astype(int)

# Encode categorical variables
le_team = LabelEncoder()
le_role = LabelEncoder()
df['TEAM_ENC'] = le_team.fit_transform(df['TEAM'])
df['ROLE_ENC'] = le_role.fit_transform(df['ROLE'])

# Features and target
X = df[['CREDITS', 'TEAM_ENC', 'ROLE_ENC']]
y = df['HIGH_SCORER']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1: Encode 'PLAYERS' just for reference (not used in features here)
label_encoder = LabelEncoder()
df['PLAYERS_encoded'] = label_encoder.fit_transform(df['PLAYERS'])

# Step 2: Define features and target
X = df[['POINTS', 'CREDITS']]
# Let's assume you're classifying whether a player is a high scorer or not
# Create a binary target: 1 if above average, else 0
average_points = df['POINTS'].mean()
y = (df['POINTS'] > average_points).astype(int)  # 1 = high scorer, 0 = not

# Step 3: Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 4: Split balanced data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Step 5: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("/Users/kaushalkhatu/Desktop/TEAM/Cleaned_DT.CSV")

# Step 1: Get model predictions (assuming model is already trained and available)
df['Predicted_Points'] = model.predict(df[['POINTS', 'CREDITS']])

# Step 2: Sort players based on predicted points (highest first)
df_sorted = df.sort_values(by='Predicted_Points', ascending=False)

# Step 3: Initialize the selected team and role counts
selected_team = []
role_count = {'WK': 0, 'AR': 0, 'BT': 0, 'BL': 0}

# Step 4: Select 2 Wicketkeepers (WK)
for _, row in df_sorted[df_sorted['ROLE'] == 'WK'].iterrows():
    if role_count['WK'] < 2:
        selected_team.append(row['PLAYERS'])
        role_count['WK'] += 1
    if len(selected_team) == 11:
        break

# Step 5: Select 3 Batsmen (BT)
for _, row in df_sorted[df_sorted['ROLE'] == 'BT'].iterrows():
    if role_count['BT'] < 3:
        selected_team.append(row['PLAYERS'])
        role_count['BT'] += 1
    if len(selected_team) == 11:
        break

# Step 6: Select 4 All-rounders (AR) or 3 if 3 bowlers are selected
for _, row in df_sorted[df_sorted['ROLE'] == 'AR'].iterrows():
    if role_count['AR'] < 4:
        selected_team.append(row['PLAYERS'])
        role_count['AR'] += 1
    if len(selected_team) == 11:
        break

# Step 7: Select 2 Bowlers (BL)
for _, row in df_sorted[df_sorted['ROLE'] == 'BL'].iterrows():
    if role_count['BL'] < 2:
        selected_team.append(row['PLAYERS'])
        role_count['BL'] += 1
    if len(selected_team) == 11:
        break

# Step 8: Ensure only 11 players are selected (no duplicates)
selected_team_df = df[df['PLAYERS'].isin(selected_team)].head(11)

# Step 9: Sort the players by role: WK, BT, AR, BL
role_order = {'WK': 1, 'BT': 2, 'AR': 3, 'BL': 4}
selected_team_df['ROLE_order'] = selected_team_df['ROLE'].map(role_order)

selected_team_df = selected_team_df.sort_values(by='ROLE_order').drop(columns=['ROLE_order'])

# Step 10: Save selected team to CSV
selected_team_df[['PLAYERS', 'TEAM', 'ROLE', 'Predicted_Points']].to_csv(
    "/Users/kaushalkhatu/Desktop/TEAM/Dream_TeamML.csv", index=False
)

# Step 11: Output selected team
print("Selected Dream11 Team:")
print(selected_team_df[['PLAYERS', 'TEAM', 'ROLE', 'Predicted_Points']])


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# Assuming `model` is already trained, and `X_test`, `y_test`, and `y_pred` are available.

# 1. Feature Importance Plot
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(features)), importances[indices], align="center")
plt.yticks(range(len(features)), features[indices])
plt.xlabel("Relative Importance")
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Not High Scorer', 'High Scorer'], yticklabels=['Not High Scorer', 'High Scorer'])
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# 4. Precision-Recall Curve (Optional for imbalance)
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.title("Precision-Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

from sklearn.model_selection import learning_curve

# Plot Learning Curves for training and validation performance
train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=5)

# Calculate mean and std deviation for plot
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training score", color="blue")
plt.plot(train_sizes, val_mean, label="Cross-validation score", color="red")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.title("Learning Curves")
plt.legend(loc="best")
plt.show()

from sklearn.model_selection import cross_val_score

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

plt.figure(figsize=(6, 6))
plt.boxplot(cv_scores, vert=False)
plt.title("Cross-Validation Scores")
plt.xlabel("Accuracy")
plt.show()

# Mean and standard deviation of cross-validation scores
print(f"Mean CV Score: {cv_scores.mean():.3f}")
print(f"CV Score Std Dev: {cv_scores.std():.3f}")

from sklearn.calibration import calibration_curve

# Get predicted probabilities
prob_pos = model.predict_proba(X_test)[:, 1]

# Calculate calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

# Plot Calibration Curve
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, marker="o", label="Model", color='blue')
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated", color="gray")
plt.xlabel("Mean Predicted Value")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend(loc="best")
plt.show()

import numpy as np

# Calculate residuals
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue', edgecolor='black', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
print(f"Best parameters found: {grid_search.best_params_}")





