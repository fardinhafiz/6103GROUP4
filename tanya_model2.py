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

#subset for just US
kickstarter1 = kickstarter1[kickstarter['country'].isin(['US'])]

#%%
# Add duration of campaign (difference between launch date and deadline)
kickstarter1['launched'] = pd.to_datetime(kickstarter1['launched']).dt.date
kickstarter1['deadline'] = pd.to_datetime(kickstarter1['deadline']).dt.date
kickstarter1['Duration'] = (pd.to_datetime(kickstarter1['deadline']) - pd.to_datetime(kickstarter1['launched'])).dt.days
print(f"Duration column added in {time.time() - start_time} seconds")

# Convert categorical columns to category type
for col in ['main_category', 'currency', 'state']:
    kickstarter1[col] = kickstarter1[col].astype('category')
print(f"Categorical columns converted in {time.time() - start_time} seconds")

# Create final dataframe with selected columns
kickstarter_final = kickstarter1[['main_category', 'currency', 'state', 'backers', 'Duration', 'usd_goal_real']]
print(f"Final dataframe created in {time.time() - start_time} seconds")

#%% 
# Add dummy variables for categorical columns
kickstarter_final = pd.get_dummies(kickstarter_final, columns=['main_category', 'currency'], drop_first=True)
print(f"Dummy variables added in {time.time() - start_time} seconds")

#%%
# Split data into features and target
X = kickstarter_final.drop(columns=['state'])
y = kickstarter_final['state']
print(f"Data split into features and target in {time.time() - start_time} seconds")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training and test sets created in {time.time() - start_time} seconds")

# Initialize the logistic regression model
model = LogisticRegression(max_iter=5000)
print(f"Logistic regression model initialized in {time.time() - start_time} seconds")

#%%
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

selected_features = ["backers", "usd_goal_real", "main_category_Comics", "main_category_Crafts", "main_category_Dance", "main_category_Design", "main_category_Fashion", "main_category_Film & Video", "main_category_Food", "main_category_Games", "main_category_Journalism", "main_category_Music", "main_category_Photography", "main_category_Publishing", "main_category_Technology", "main_category_Theater", "Duration"]

X = kickstarter_final[selected_features]
y = kickstarter_final['state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 251)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fitting the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Accuracy
print('Logit model accuracy (test set):', model.score(X_test, y_test))
print('Logit model accuracy (train set):', model.score(X_train, y_train))

# Coefficients and Odds Ratios for features
coefficients = pd.DataFrame({
    'Predictors': X.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
})
print("\nCoefficients and Odds Ratios:\n", coefficients)

# Predictions and Evaluation
y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n The confusion matrix of the model is:")
print(conf_matrix)

print("\n The accuracy of the model is:")
print(accuracy)

print("\n The model's classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve and AUC
y_test_binary = [1 if label == 'successful' else 0 for label in y_test]  # Binary conversion

# Get the probability scores for the positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)
roc_auc = auc(fpr, tpr)  # Calculate AUC

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.3)
plt.text(0.4, 0.5, f'AUC = {roc_auc:.3f}', fontsize=12)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

#%%
#try again
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # optional
import statsmodels.api as sm

start_time = time.time()
print("Starting script execution...")

# Read the data
kickstarter = pd.read_csv("ks-projects-201801.csv")
print(f"Data read in {time.time() - start_time} seconds")

# Subset for just failed or success
kickstarter1 = kickstarter[kickstarter['state'].isin(['failed', 'successful'])]
print(f"Subset created in {time.time() - start_time} seconds")

# Subset for just US
kickstarter1 = kickstarter1[kickstarter['country'] == 'US']

# Add duration of campaign (difference between launch date and deadline)
kickstarter1['launched'] = pd.to_datetime(kickstarter1['launched']).dt.date
kickstarter1['deadline'] = pd.to_datetime(kickstarter1['deadline']).dt.date
kickstarter1['Duration'] = (pd.to_datetime(kickstarter1['deadline']) - pd.to_datetime(kickstarter1['launched'])).dt.days
print(f"Duration column added in {time.time() - start_time} seconds")

# Convert categorical columns to category type
for col in ['main_category', 'currency', 'state']:
    kickstarter1[col] = kickstarter1[col].astype('category')
print(f"Categorical columns converted in {time.time() - start_time} seconds")

# Create final dataframe with selected columns
kickstarter_final = kickstarter1[['main_category', 'currency', 'state', 'backers', 'Duration', 'usd_goal_real']]
print(f"Final dataframe created in {time.time() - start_time} seconds")

# Add dummy variables for categorical columns
kickstarter_final = pd.get_dummies(kickstarter_final, columns=['main_category', 'currency'], drop_first=True)
print(f"Dummy variables added in {time.time() - start_time} seconds")

# Split data into features and target
X = kickstarter_final.drop(columns=['state'])
y = kickstarter_final['state'].cat.codes  # Convert categories to numerical codes for the target variable
print(f"Data split into features and target in {time.time() - start_time} seconds")

# Ensure all data is numeric
X = X.apply(pd.to_numeric)
y = pd.to_numeric(y)

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the logistic regression model using statsmodels
model = sm.Logit(y, X).fit()
print(model.summary())

# Model summary provides coefficients, p-values, and other statistical measures
coefficients = pd.DataFrame({
    'Predictors': X.columns,
    'Coefficient': model.params,
    'Odds Ratio': np.exp(model.params),
    'P-value': model.pvalues
})
print("\nCoefficients, Odds Ratios, and P-values:\n", coefficients)

# Predictions and evaluation
y_pred = model.predict(X)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

conf_matrix = pd.crosstab(y, np.array(y_pred_binary), rownames=['Actual'], colnames=['Predicted'])
accuracy = (y == y_pred_binary).mean()

print("\nThe confusion matrix of the model is:")
print(conf_matrix)

print("\nThe accuracy of the model is:")
print(accuracy)

# ROC Curve and AUC
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.3)
plt.text(0.4, 0.5, f'AUC = {roc_auc:.3f}', fontsize=12)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# %%
