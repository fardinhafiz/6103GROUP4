
#%%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

start_time = time.time()
print("Starting script execution...")

# Read the data
kickstarter = pd.read_csv("ks-projects-201801.csv")
print(f"Data read in {time.time() - start_time} seconds")

# Subset for just failed or success
kickstarter1 = kickstarter[kickstarter['state'].isin(['failed', 'successful'])]
print(f"Subset created in {time.time() - start_time} seconds")

# Add new feature of percent of goal reached 
kickstarter1['percent_funded'] = kickstarter["usd_pledged_real"]/kickstarter['usd_goal_real']
print(kickstarter1.head())

#%%
# Add duration of campaign (difference between launch date and deadline)
kickstarter1['launched'] = pd.to_datetime(kickstarter1['launched']).dt.date
kickstarter1['deadline'] = pd.to_datetime(kickstarter1['deadline']).dt.date
kickstarter1['Duration'] = (pd.to_datetime(kickstarter1['deadline']) - pd.to_datetime(kickstarter1['launched'])).dt.days
print(f"Duration column added in {time.time() - start_time} seconds")

# Convert categorical columns to category type
for col in ['main_category', 'currency', 'state', 'country']:
    kickstarter1[col] = kickstarter1[col].astype('category')
print(f"Categorical columns converted in {time.time() - start_time} seconds")

# Create final dataframe with selected columns
kickstarter_final = kickstarter1[['main_category', 'currency', 'state', 'backers', 'country', 'percent_funded', 'Duration', 'usd_goal_real']]
print(f"Final dataframe created in {time.time() - start_time} seconds")

#%% 

# # change state to int

# # Map the values
# kickstarter_final['state'] = kickstarter_final['state'].map({'failed': 0, 'successful': 1})

# Add dummy variables for categorical columns
kickstarter_final = pd.get_dummies(kickstarter_final, columns=['main_category', 'currency', 'country'])
print(f"Dummy variables added in {time.time() - start_time} seconds")


# corrmatrix = kickstarter_final.corr()

# plt.figure(figsize=(8, 6))
# sns.heatmap(corrmatrix, annot=False, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Matrix Heatmap')
# plt.show()


#%%
# Split data into features and target
X = kickstarter_final.drop(columns=['state'])
y = kickstarter_final['state']
print(f"Data split into features and target in {time.time() - start_time} seconds")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training and test sets created in {time.time() - start_time} seconds")

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)
print(f"Logistic regression model initialized in {time.time() - start_time} seconds")

#%%
# Forward feature selection
remaining_features = list(X_train.columns)
selected_features = []
best_score = 0

while remaining_features:
    scores_with_candidates = []
    for candidate in remaining_features:
        candidate_features = selected_features + [candidate]
        model.fit(X_train[candidate_features], y_train)
        y_pred = model.predict(X_test[candidate_features])
        score = accuracy_score(y_test, y_pred)
        scores_with_candidates.append((score, candidate))
    scores_with_candidates.sort()
    best_new_score, best_candidate = scores_with_candidates.pop()

    if best_new_score > best_score:
        remaining_features.remove(best_candidate)
        selected_features.append(best_candidate)
        best_score = best_new_score
    else:
        break
    print(f"Forward selection: selected {best_candidate} with score {best_new_score} at {time.time() - start_time} seconds")

print(f"Selected features: {selected_features}")
print(f"Best score: {best_score}")

# Train the final model with selected features
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train[selected_features], y_train)
final_predictions = final_model.predict(X_test[selected_features])
print(f"Final model trained in {time.time() - start_time} seconds")

# Final model accuracy
final_accuracy = accuracy_score(y_test, final_predictions)
print(f"Final model accuracy: {final_accuracy}")
print(f"Script finished in {time.time() - start_time} seconds")

# %%
