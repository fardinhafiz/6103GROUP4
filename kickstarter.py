
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import matplotlib.colors as mcolors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from statsmodels.formula.api import glm
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import time
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import export_text


#%%
# read in the dataset 

kickstarter = pd.read_csv("ks-projects-201801.csv")

print("\nReady to continue.")
# %%
# first 5 rows 
kickstarter.head()
# %%
# check datatypes 
kickstarter.info()

# check unique values for 'state' and 'main_category'
kickstarter['state'].unique()
kickstarter['main_category'].unique()

# %%

# subset for just failed or success, reduces set to 331675 rows with 15 variables

kickstarter1 = kickstarter[kickstarter['state'].isin(['failed', 'successful'])]
print(kickstarter1)

# %%

#add duration of campain (difference between launch date and deadline)
kickstarter1['launched'] = pd.to_datetime(kickstarter1['launched']).dt.date

# Convert 'deadline' to datetime and then to date (date-only)
kickstarter1['deadline'] = pd.to_datetime(kickstarter1['deadline']).dt.date

# Calculate the duration (difference in days) between deadline and launched
kickstarter1['Duration'] = (pd.to_datetime(kickstarter1['deadline']) - pd.to_datetime(kickstarter1['launched'])).dt.days

# Display the DataFrame
print(kickstarter1[['launched', 'deadline', 'Duration']])
# %%

# change the objects to factors
kickstarter1['main_category'] = kickstarter1['main_category'].astype('category')
kickstarter1['currency'] = kickstarter1['currency'].astype('category')
kickstarter1['state'] = kickstarter1['state'].astype('category')
kickstarter1['country'] = kickstarter1['country'].astype('category')

kickstarter1.info()
# %%
# subset with just main_category, currency, state, backers, country,
# usd_pledged_real, usd_goal_real, Duration

kickstarter_final = kickstarter1[['main_category', 'currency', 'state', 'backers', 'country', 'usd_pledged_real', 'usd_goal_real', 'Duration']]
print(kickstarter_final)

# %%
# summary stats (for all countries)
# Describe continuous variables
print(kickstarter_final[['backers', 'usd_goal_real', 'usd_pledged_real', 'Duration']].describe())

# Describe categorical variables
print(kickstarter_final[['main_category', 'state', 'currency', 'country']].apply(lambda x: x.describe(include='all')).T)

#%% 
# correlation plot for dataframe for all countries 
# One-hot encode the categorical variables
kickstarter_final_encoded = pd.get_dummies(kickstarter_final, drop_first=True)

# Calculate the correlation matrix
kickstarter_final_encoded_corr_matrix = kickstarter_final_encoded.corr()

# Generate a heat map
plt.figure(figsize=(10, 8))
sns.heatmap(kickstarter_final_encoded_corr_matrix, annot=False, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix Heat Map')
plt.show()

#%%
# distribution of failed vs success

distribution = kickstarter_final['state'].value_counts()
print(distribution)
percentage_distribution = kickstarter_final['state'].value_counts(normalize=True) * 100
print(percentage_distribution)

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
kickstarter_final['state'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Distribution of Success vs Failure')
plt.xticks(rotation = 0)
plt.ylabel('Count')

plt.subplot(1,2,2)
plt.pie(
    distribution, 
    labels=distribution.index, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=['red', 'green'])
plt.title('Distribution of Kickstarter Project Outcomes')
plt.show()

#%% 
# Filter the dataframe for successful and failed projects
successful_projects = kickstarter_final[kickstarter_final['state'] == "successful"]
failed_projects = kickstarter_final[kickstarter_final['state'] == "failed"]

# Create the scatter plot for successful projects
plt.figure(figsize=(12, 8))
sns.scatterplot(data=successful_projects, 
                x='usd_goal_real', 
                y='backers', 
                hue='main_category', 
                palette='Set2')
plt.title('Successful Kickstarter Projects: Goal Amount vs. Number of Backers by Main Category')
plt.xlabel('Goal Amount in USD')
plt.ylabel('Number of Backers')
plt.legend(title='Main Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the first plot
plt.show()

# Create the scatter plot for failed projects
plt.figure(figsize=(12, 8))
sns.scatterplot(data=failed_projects, 
                x='usd_goal_real', 
                y='backers', 
                hue='main_category', 
                palette='Set2')
plt.title('Failed Kickstarter Projects: Goal Amount vs. Number of Backers by Main Category')
plt.xlabel('Goal Amount in USD')
plt.ylabel('Number of Backers')
plt.legend(title='Main Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the second plot
plt.show()

# %%

# state by country, currency, and category
grouped_country = kickstarter_final.groupby(['country', 'state']).size().unstack()

grouped_country.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'green'])
plt.title('Projects by Country, and Outcome (Stacked)')
plt.ylabel('Count')
plt.xlabel('Category')
plt.legend(title='Outcome')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

grouped_currency = kickstarter_final.groupby(['currency', 'state']).size().unstack()

grouped_currency.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'green'])
plt.title('Projects by Currency, and Outcome (Stacked)')
plt.ylabel('Count')
plt.xlabel('Category')
plt.legend(title='Outcome')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

grouped_category = kickstarter_final.groupby(['main_category', 'state']).size().unstack(fill_value=0)

grouped_category.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'green'])
plt.title('Projects by Category, and Outcome (Stacked)')
plt.ylabel('Count')
plt.xlabel('Category')
plt.legend(title='Outcome')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%% 
# top 5 categories (based on percentage of successful projects in the category)

total_projects_per_category = kickstarter_final.groupby('main_category').size()
successful_projects_per_category = kickstarter_final[kickstarter_final['state'] == 'successful'].groupby('main_category').size()

# Calculate the percentage of successful projects
success_percentage = (successful_projects_per_category / total_projects_per_category) * 100

# Get the top categories with the highest percentage of successful projects
top_categories_percentage = success_percentage.sort_values(ascending=False).head(5)

print("Top 5 categories with the Highest Percentage of Successful Projects (all countries)")
print(top_categories_percentage)

#%% 
# using median goal instead of mean goal

# Calculate the total number of projects per category
total_projects_per_category_all = kickstarter_final.groupby('main_category').size()

# Calculate the total number of successful projects per category
successful_projects_per_category_all = kickstarter_final[kickstarter_final['state'] == 'successful'].groupby('main_category').size()

# Calculate the percentage of successful projects
success_percentage_all = (successful_projects_per_category_all / total_projects_per_category_all) * 100

# Sort the categories by their success percentages
sorted_success_percentage_all = success_percentage_all.sort_values(ascending=False)

print("Percentage of Successful Projects per Category")
print(sorted_success_percentage_all)

# Median goal per category 
median_goal_per_category = kickstarter_final.groupby('main_category')['usd_goal_real'].median()

# Normalize funding goals for color mapping
norm = mcolors.Normalize(vmin=median_goal_per_category.min(), vmax=median_goal_per_category.max())
colors = plt.cm.coolwarm(norm(median_goal_per_category[sorted_success_percentage_all.index]))

fig, ax = plt.subplots(figsize=(12, 8))
sorted_success_percentage_all.plot(kind='bar', color=colors, ax=ax)
plt.title('Percentage of Successful Projects in Each Category (all countries included)', fontsize=14)
plt.ylabel('Percent of Successful Projects (%)')
plt.xlabel('Main Category')
plt.xticks(rotation=45, ha='right', fontsize=10)

# Adding the values onto each bar
for index, value in enumerate(sorted_success_percentage_all):
    ax.text(index, value + 0.5, f'{round(value, 2)}%', ha='center', va='bottom', fontsize=10)

# Create a colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Median Goal (USD)')

plt.tight_layout()
plt.show()

#%% 
# top 5 categories percentage (US only)

us_kickstarter_final = kickstarter_final[kickstarter_final['country'] == 'US']

total_projects_per_category_us = us_kickstarter_final.groupby('main_category').size()
successful_projects_per_category_us = us_kickstarter_final[kickstarter_final['state'] == 'successful'].groupby('main_category').size()

# Calculate the percentage of successful projects
success_percentage_us = (successful_projects_per_category_us / total_projects_per_category_us) * 100

# Get the top categories with the highest percentage of successful projects
top_categories_percentage_us = success_percentage_us.sort_values(ascending=False).head(5)

print("Top 5 categories with the Highest Percentage of Successful Projects (all countries)")
print(top_categories_percentage_us)

# using median goal instead of mean goal
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Calculate the total number of projects per category
total_projects_us = us_kickstarter_final.groupby('main_category').size()

# Calculate the total number of successful projects per category
successful_projects_us = us_kickstarter_final[us_kickstarter_final['state'] == 'successful'].groupby('main_category').size()

# Calculate the percentage of successful projects
successful_projects_us = (successful_projects_us / total_projects_us) * 100

# Sort the categories by their success percentages
sorted_success_percentage_us = successful_projects_us.sort_values(ascending=False)

print("Percentage of Successful Projects per Category")
print(sorted_success_percentage_us)

# Median goal per category 
median_goal_per_category_us = us_kickstarter_final.groupby('main_category')['usd_goal_real'].median()

# Normalize funding goals for color mapping
norm = mcolors.Normalize(vmin=median_goal_per_category_us.min(), vmax=median_goal_per_category_us.max())
colors = plt.cm.coolwarm(norm(median_goal_per_category_us[sorted_success_percentage_us.index]))

fig, ax = plt.subplots(figsize=(12, 8))
sorted_success_percentage_us.plot(kind='bar', color=colors, ax=ax)
plt.title('Percentage of Successful Projects in Each Category (US only)', fontsize=14)
plt.ylabel('Percent of Successful Projects (%)')
plt.xlabel('Main Category')
plt.xticks(rotation=45, ha='right', fontsize=10)

# Adding the values onto each bar
for index, value in enumerate(sorted_success_percentage_us):
    ax.text(index, value + 0.5, f'{round(value, 2)}%', ha='center', va='bottom', fontsize=10)

# Create a colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Median Goal (USD)')

plt.tight_layout()
plt.show()

#%%
#success and failure by backers and funding goal

plt.figure(figsize=(6, 10))  # Larger plot for better visibility

sns.scatterplot(
    data=kickstarter_final,
    x='state',
    y='backers',
    size='usd_goal_real',  # Map `use_goal_real` to marker size
    hue='usd_goal_real',  # Map `use_goal_real` to color
    palette='viridis',  # A perceptually uniform colormap
    sizes=(50, 500),  # Adjust marker sizes
    alpha=0.7  # Add some transparency to reduce overlap
)

plt.title('State by Number of Backers and Funding Goal', fontsize=20)
plt.xlabel('Campaign State', fontsize=14)
plt.ylabel('Number of Backers', fontsize=14)
plt.yscale('log')  # Optional: Use log scale if needed
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Goal (USD)', fontsize=12, title_fontsize=14, loc='center')
plt.tight_layout()
plt.show()

#%% 


# Separate the data based on state
successful_goals = kickstarter_final[kickstarter_final['state'] == 'successful']['usd_goal_real']
failed_goals = kickstarter_final[kickstarter_final['state'] == 'failed']['usd_goal_real']

# Plot histogram for successful goals
plt.figure(figsize=(12, 6))
plt.hist(successful_goals, bins = 5000, alpha=0.7, color='blue')
plt.xlabel('Goal Amount (USD)')
plt.ylabel('Frequency')
plt.title('Histogram of Goal Amounts for Successful Projects')
plt.xlim(0, 20000)
plt.show()

# Plot histogram for failed goals
plt.figure(figsize=(12, 6))
plt.hist(failed_goals, alpha=0.7, color='red')
plt.xlabel('Goal Amount (USD)')
plt.ylabel('Frequency')
plt.title('Histogram of Goal Amounts for Failed Projects')
plt.show()

#%%

plt.figure(figsize=(10, 6))
sns.scatterplot(data=kickstarter_final, x='usd_goal_real', y='backers', hue='state', palette='viridis', alpha=0.7)

# Customizing the plot
plt.title('Scatterplot of Backers vs Goal Colored by State')
plt.xlabel('Goal (use_goal_real)')
plt.ylabel('Backers')
plt.legend(title='State', loc='upper right')
plt.grid(True)

# Show the plot
plt.show()

# %%
# create a training set

train_set, test_set = train_test_split(kickstarter_final, train_size=800, random_state=42)

#%%
#fit tree to training data

X_trainkickstarter = train_set.drop(columns=['state'])
y_trainkickstarter = train_set['state']

X_trainkickstarter = pd.get_dummies(X_trainkickstarter, drop_first=True)

dtree_kickstarter = DecisionTreeClassifier(max_depth = 4, criterion = 'gini', random_state = 1)

dtree_kickstarter.fit(X_trainkickstarter, y_trainkickstarter)

y_trainkickstarter_pred = dtree_kickstarter.predict(X_trainkickstarter)


training_error_rate_kickstarter = 1 - accuracy_score(y_trainkickstarter, y_trainkickstarter_pred)

print(f"Training error rate: {training_error_rate_kickstarter:.4f}")

#%%
# Comparison of cross validations to find best depth
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_rel
import pandas as pd

# Assuming you have already loaded your data and done the necessary preprocessing
X_trainkickstarter = train_set.drop(columns=['state'])
y_trainkickstarter = train_set['state']

X_trainkickstarter = pd.get_dummies(X_trainkickstarter, drop_first=True)

# Cross-validation for max_depth=3
dtree_depth3 = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=1)
scores_depth3 = cross_val_score(dtree_depth3, X_trainkickstarter, y_trainkickstarter, cv=5)

# Cross-validation for max_depth=4
dtree_depth4 = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=1)
scores_depth4 = cross_val_score(dtree_depth4, X_trainkickstarter, y_trainkickstarter, cv=5)

dtree_depth5 = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=1)
scores_depth5 = cross_val_score(dtree_depth5, X_trainkickstarter, y_trainkickstarter, cv=5)

t_stat, p_value = ttest_rel(scores_depth3, scores_depth4)

print(f"Scores for max_depth=3: {scores_depth3}")
print(f"Scores for max_depth=4: {scores_depth4}")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Check if the difference is significant (typically, p < 0.05)
if p_value < 0.05:
    print("The difference in performance is statistically significant.")
else:
    print("The difference in performance is not statistically significant.")

# Perform paired t-test
t_stat, p_value = ttest_rel(scores_depth4, scores_depth5)

print(f"Scores for max_depth=4: {scores_depth4}")
print(f"Scores for max_depth=5: {scores_depth5}")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Check if the difference is significant (typically, p < 0.05)
if p_value < 0.05:
    print("The difference in performance is statistically significant.")
else:
    print("The difference in performance is not statistically significant.")

#%%
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np

# Define parameter range
param_range = np.arange(1, 15)

# Calculate training and validation scores
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(random_state=1),
    X_trainkickstarter, y_trainkickstarter,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy"
)

# Calculate mean and standard deviation for training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curve
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray", alpha=0.2)
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gray", alpha=0.2)

plt.title("Validation Curve with Decision Tree")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()


#%%

#plot the tree

plt.figure(figsize=(12,8))
plot_tree(dtree_kickstarter, feature_names=X_trainkickstarter.columns, class_names=dtree_kickstarter.classes_, filled=True, rounded=True)
plt.title('Decision Tree for Kickstarter Campaign Outcomes')
plt.show()

n_terminal_nodes = sum(dtree_kickstarter.tree_.children_left == -1)
print(f"Number of terminal nodes (leaf nodes): {n_terminal_nodes}")

#%%

# Generate a text summary of the tree
tree_rules = export_text(dtree_kickstarter, feature_names=X_trainkickstarter.columns.tolist())
print(tree_rules)

#%%

#predict response on the test data and produce confusion matrix



X_testkickstarter = pd.get_dummies(test_set.drop(columns=['state']), drop_first=True)

# Align test set columns with training set columns
X_testkickstarter = X_testkickstarter.reindex(columns=X_trainkickstarter.columns, fill_value=0)

y_testkickstarter = test_set['state']


y_testkickstarter_pred = dtree_kickstarter.predict(X_testkickstarter)

conf_matrix = confusion_matrix(y_testkickstarter, y_testkickstarter_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=dtree_kickstarter.classes_, yticklabels=dtree_kickstarter.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Test Set')
plt.show()

test_error_rate = 1 - accuracy_score(y_testkickstarter, y_testkickstarter_pred)
print(f"Test error rate: {test_error_rate:.4f}")

# %%

maxlevels = [None, 2, 3, 5, 8]
crits = ['gini', 'entropy']
for l in maxlevels:
    for c in crits:
        dt = DecisionTreeClassifier(max_depth = l, criterion = c)
        dt.fit(X_trainkickstarter, y_trainkickstarter)
        print(l, c, dt.score(X_testkickstarter, y_testkickstarter))

# %%

# Logistic Regression Model
# Features selected: backers, usd_pledged_real, main_category

X = pd.get_dummies(kickstarter_final[['backers', 'usd_pledged_real', 'main_category']], drop_first=True)
y = (kickstarter_final['state'] == 'successful').astype(int) 

# Train-Test Split; 70:30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 251)

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

#%%
TN, FP, FN, TP = conf_matrix.ravel() #Obtaining values from confusion matrix

# Calculating FPR and FNR
FPR = FP / (FP + TN)  # False Positive Rate
FNR = FN / (FN + TP)  # False Negative Rate

print(f"False Positive Rate (FPR): {FPR*100:.2f}%")
print(f"False Negative Rate (FNR): {FNR*100:.2f}%")


#%%
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False, annot_kws={"size": 14}) # add color, size, etc
plt.title("Confusion Matrix", fontsize = 14)
plt.xlabel("Predicted", fontsize = 12)
plt.ylabel("True", fontsize = 12)
plt.show()

#%%
from sklearn import metrics

y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
roc_auc = roc_auc_score(y_test, y_prob)  # AUC score

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange', lw=2)
plt.fill_between(fpr, tpr, color='lightcoral', alpha=0.5)  # Filling in AUC by shading
plt.plot([0,1],[0,1],color='black', linestyle='--', lw=1, alpha = 0.7)  # Diagonal line

# Customize the plot
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(fontsize=12, loc=[0.5, 0.1])
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#%%

#New Model, different threshold

threshold = 0.3
y_pred_adjusted = (y_prob >= threshold).astype(int)

# Confusion Matrix and Metrics with adjusted threshold
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
print("\nConfusion Matrix with Threshold 0.3:\n", conf_matrix_adjusted)

accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
print(f"\nAccuracy with Threshold 0.3: {accuracy_adjusted:.2f}")

print("\nClassification Report with Threshold 0.3:")
print(classification_report(y_test, y_pred_adjusted))

#%%
#Confusion Matrix with Threshold of 0.3:

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_adjusted, annot=True, fmt='d', cmap='viridis', cbar=False, annot_kws={"size": 14})
plt.title("Confusion Matrix with threshold 0.3", fontsize = 14)
plt.xlabel("Predicted", fontsize = 12)
plt.ylabel("True", fontsize = 12)
plt.show()

#%%
new_TN, new_FP, new_FN, new_TP = conf_matrix_adjusted.ravel()

# Calculating new FPR and FNR at threshold = 0.3
new_FPR = new_FP / (new_FP + new_TN)  # False Positive Rate
new_FNR = new_FN / (new_FN + new_TP)  # False Negative Rate

print(f"False Positive Rate (FPR) at threshold 0.3: {new_FPR*100:.2f}%")
print(f"False Negative Rate (FNR) at threshold 0.3: {new_FNR*100:.2f}%")

# %%

# Creating ROC curve with AUC for logistic model

y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob) # evaluating AUC

print(f"\n The area under the curve is found to be {roc_auc:.3f}.")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

# Shading the AUC
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.3)

# Text with AUC value inside the plot
plt.text(0.4, 0.5, f'AUC = {roc_auc:.3f}', fontsize=12)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# %%
# %%
# prep US only stuff 

# Create final dataframe with selected columns
kickstarter_final_US = kickstarter_final.copy()

#subset out US 
kickstarter_final_US = kickstarter_final_US[kickstarter_final_US['country'] == 'US']

#rename film category 
kickstarter_final_US['main_category'] = kickstarter_final_US['main_category'].replace({'Film & Video': 'Film_and_Video'})

kickstarter_final_US = kickstarter_final_US.drop(['currency', 'country'], axis = 1)

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

#binary state
kickstarter_us_binary = kickstarter_final_US.copy()
kickstarter_us_binary['state_binary'] = kickstarter_us_binary['state'].map({'failed': 0, 'successful': 1})
kickstarter_us_binary['state_binary'] = kickstarter_us_binary['state_binary'].astype(int)

kickstarter_us_binary = kickstarter_us_binary.drop(['state'], axis = 1)

# Dummy variables
kickstarter_us_binary = pd.get_dummies(kickstarter_us_binary, columns=['main_category'], drop_first=True)

# Sample a smaller subset (e.g., 5% of the data) for faster feature selection
def sample_data(df, frac=0.05, random_state=42):
    return kickstarter_us_binary.sample(frac=frac, random_state=random_state)

# Helper function to print timings
def print_timing(message, start_time):
    print(f'{message}: {time.time() - start_time:.2f} seconds')

# Prepare the data
train_df_select, test_df_select = train_test_split(kickstarter_us_binary, test_size=0.2, random_state=42)
x_us_select = train_df_select.drop(columns=['state_binary'], axis=1)
y_us_select = train_df_select['state_binary']

# Sample 5% of the training data for feature selection
sampled_train_df = sample_data(train_df_select, frac=0.05)
x_sampled_us_select = sampled_train_df.drop(columns=['state_binary'], axis=1)
y_sampled_us_select = sampled_train_df['state_binary']

logistic_model = LogisticRegression(max_iter=5000)

print("Starting feature selection...")
start_time = time.time()

# Perform forward feature selection on the sampled data with reduced cross-validation folds
sfs_us = SFS(logistic_model,
             k_features='best',
             forward=True,
             floating=False,
             scoring='accuracy',
             cv=3,  # Reduced number of CV folds
             n_jobs=-1)  # Use all available cores
sfs_us = sfs_us.fit(x_sampled_us_select, y_sampled_us_select)

selection_time = time.time() - start_time
print_timing("Feature selection time", start_time)
selected_features_us = list(sfs_us.k_feature_names_)
print(f'Selected features: {selected_features_us}')

print("Starting model fitting with full data...")
start_time = time.time()
x_selected_us_features = x_us_select[selected_features_us]
logistic_model.fit(x_selected_us_features, y_us_select)
fitting_time = time.time() - start_time
print_timing("Model fitting time", start_time)

print("Starting model evaluation...")
start_time = time.time()
x_test_us_select = test_df_select[selected_features_us]
y_test_us_select = test_df_select['state_binary']
y_pred_us_select = logistic_model.predict(x_test_us_select)
accuracy_us_select = accuracy_score(y_test_us_select, y_pred_us_select)
print(f'Accuracy: {accuracy_us_select}')
evaluation_time = time.time() - start_time
print_timing("Model evaluation time", start_time)

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Convert boolean columns to integers
def convert_boolean_to_int(data):
    bool_cols = data.select_dtypes(include=['bool']).columns
    data[bool_cols] = data[bool_cols].astype(int)
    return data

# Calculating VIF with NaN, Infinity checks, and boolean conversion
def calculate_vif(data):
    # Convert boolean columns to integers
    data = convert_boolean_to_int(data)

    # Convert data to numeric and handle errors
    data = data.apply(pd.to_numeric, errors='coerce')

    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    return vif_data

# Running VIF on the selected features
x_selected_us_features_with_vif = x_us_select[selected_features_us]
vif_df = calculate_vif(x_selected_us_features_with_vif)
print(vif_df)

# %%
#try us only 

train_df_stats, test_df_stats = train_test_split(kickstarter_us_binary, test_size=0.2, random_state=42)

# Define the formula for logistic regression
formula = 'state_binary ~ backers + usd_pledged_real + main_category_Comics + main_category_Crafts + main_category_Dance + main_category_Design + main_category_Fashion + main_category_Film_and_Video + main_category_Food + main_category_Games + main_category_Journalism + main_category_Music + main_category_Photography + main_category_Publishing + main_category_Technology + main_category_Theater'

# Fit the logistic regression model
stats_model_us = smf.logit(formula, data=train_df_stats).fit()

# Make predictions on the test data
test_df_stats['predicted'] = stats_model_us.predict(test_df_stats)
test_df_stats['predicted_class'] = (test_df_stats['predicted'] > 0.5).astype(int)

# Display the model summary
print(stats_model_us.summary())

# Extract the independent variables from the training data
independent_vars_for_vif = train_df_stats[['backers', 'usd_pledged_real', 'main_category_Comics', 'main_category_Crafts', 
                                   'main_category_Dance', 'main_category_Design', 'main_category_Fashion', 
                                   'main_category_Film_and_Video', 'main_category_Food', 'main_category_Games', 
                                   'main_category_Journalism', 'main_category_Music', 'main_category_Photography', 
                                   'main_category_Publishing', 'main_category_Technology', 'main_category_Theater']]

# Convert boolean variables to integers
independent_vars_for_vif = independent_vars_for_vif.applymap(lambda x: int(x) if isinstance(x, bool) else x)

# Ensure all data types are numeric
independent_vars_for_vif = independent_vars_for_vif.apply(pd.to_numeric, errors='coerce')

# Create a DataFrame to store VIF values
vif_data = pd.DataFrame()
vif_data['Feature'] = independent_vars_for_vif.columns
vif_data['VIF'] = [variance_inflation_factor(independent_vars_for_vif.values, i) for i in range(len(independent_vars_for_vif.columns))]

print(vif_data)

# Evaluate the model's performance on the test data
accuracy_stats_test = accuracy_score(test_df_stats['state_binary'], test_df_stats['predicted_class'])
conf_matrix_stats_test = confusion_matrix(test_df_stats['state_binary'], test_df_stats['predicted_class'])
class_report_stats_test = classification_report(test_df_stats['state_binary'], test_df_stats['predicted_class'])

print(f'Test Accuracy: {accuracy_stats_test}')
print('Test Confusion Matrix:')
print(conf_matrix_stats_test)
print('Test Classification Report:')
print(class_report_stats_test)

# Evaluate the model's performance on the training data
train_df_stats['predicted'] = stats_model_us.predict(train_df_stats)
train_df_stats['predicted_class'] = (train_df_stats['predicted'] > 0.5).astype(int)

accuracy_stats_train = accuracy_score(train_df_stats['state_binary'], train_df_stats['predicted_class'])
conf_matrix_stats_train = confusion_matrix(train_df_stats['state_binary'], train_df_stats['predicted_class'])
class_report_stats_train = classification_report(train_df_stats['state_binary'], train_df_stats['predicted_class'])

print(f'Training Accuracy: {accuracy_stats_train}')
print('Training Confusion Matrix:')
print(conf_matrix_stats_train)
print('Training Classification Report:')
print(class_report_stats_train)
#%% 

# take sample of dataset and do feature selection
# fit the model to full dataset

# Sample a smaller subset (e.g., 5% of the data) for faster feature selection
def sample_data(df, frac=0.05, random_state=42):
    return kickstarter_final_US.sample(frac=frac, random_state=random_state)

# Helper function to print timings
def print_timing(message, start_time):
    print(f'{message}: {time.time() - start_time:.2f} seconds')

# Prepare the data
train_df_select, test_df_select = train_test_split(kickstarter_final_US, test_size=0.2, random_state=42)
x_us_select = train_df_select.drop(columns=['state_binary', 'state'], axis=1)
y_us_select = train_df_select['state_binary']

# Sample 5% of the training data for feature selection
sampled_train_df = sample_data(train_df_select, frac=0.05)
x_sampled_us_select = sampled_train_df.drop(columns=['state_binary', 'state'], axis=1)
y_sampled_us_select = sampled_train_df['state_binary']

logistic_model = LogisticRegression(max_iter=5000)

print("Starting feature selection...")
start_time = time.time()

# Perform forward feature selection on the sampled data with reduced cross-validation folds
sfs_us = SFS(logistic_model,
             k_features='best',
             forward=True,
             floating=False,
             scoring='accuracy',
             cv=3,  # Reduced number of CV folds
             n_jobs=-1)  # Use all available cores
sfs_us = sfs_us.fit(x_sampled_us_select, y_sampled_us_select)

selection_time = time.time() - start_time
print_timing("Feature selection time", start_time)
selected_features_us = list(sfs_us.k_feature_names_)
print(f'Selected features: {selected_features_us}')

print("Starting model fitting with full data...")
start_time = time.time()
x_selected_us_features = x_us_select[selected_features_us]
logistic_model.fit(x_selected_us_features, y_us_select)
fitting_time = time.time() - start_time
print_timing("Model fitting time", start_time)

print("Starting model evaluation...")
start_time = time.time()
x_test_us_select = test_df_select[selected_features_us]
y_test_us_select = test_df_select['state_binary']
y_pred_us_select = logistic_model.predict(x_test_us_select)
accuracy_us_select = accuracy_score(y_test_us_select, y_pred_us_select)
print(f'Accuracy: {accuracy_us_select}')
evaluation_time = time.time() - start_time
print_timing("Model evaluation time", start_time)

# %%
# Average % met and shows the inconsistency in the data with the cancelled state
data = kickstarter[kickstarter['state'].isin(['failed', 'canceled', 'successful'])]

canceled_inconsistent = data[(data['state'] == 'canceled') & (data['usd_pledged_real'] > data['usd_goal_real'])]
num_removed_canceled = canceled_inconsistent.shape[0]

total_canceled = data[data['state'] == 'canceled'].shape[0]
num_kept_canceled = total_canceled - num_removed_canceled

print(f"Number of logically inconsistent 'canceled' rows removed: {num_removed_canceled}")
print(f"Number of 'canceled' rows kept: {num_kept_canceled}")

data_clean = data[~((data['state'] == 'canceled') & (data['usd_pledged_real'] > data['usd_goal_real']))]
data_clean['percentage_met'] = (data_clean['usd_pledged_real'] / data_clean['usd_goal_real']) * 100

data_filtered = data_clean[data_clean['state'].isin(['failed', 'successful'])]

goal_met_percentage = data_filtered.groupby('state')['percentage_met'].mean().reset_index()

plt.figure(figsize=(8, 6))
plt.bar(goal_met_percentage['state'], goal_met_percentage['percentage_met'], color=['red', 'green'])
plt.title('Average Percentange Met of Campaign Goal')
plt.ylabel('Average Percentage of Goal Met (%)')
plt.xlabel('Project State')
plt.ylim(0, max(goal_met_percentage['percentage_met']) + 20)
for index, value in enumerate(goal_met_percentage['percentage_met']):
    plt.text(index, value + 2, f"{value:.2f}%", ha='center')
plt.show()

plt.figure(figsize=(6, 6))
plt.pie([num_removed_canceled, num_kept_canceled],
        labels=['Removed Canceled Rows', 'Kept Canceled Rows'],
        autopct='%1.1f%%', startangle=90, colors=['red', 'orange'])
plt.title('Breakdown of Removed vs. Kept Canceled Rows')
plt.show()

# Zero backers
zero_backers_count = kickstarter[kickstarter['backers'] == 0].shape[0]
one_or_more_backers_count = kickstarter[kickstarter['backers'] >= 1].shape[0]

labels = ['Zero Backers', '1+ Backers']
sizes = [zero_backers_count, one_or_more_backers_count]
colors = ['red', 'green']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
plt.title('Distribution of Projects: Zero Backers vs 1+ Backers')
plt.show()


# %%
# Histogram of Backers
backers = kickstarter['backers']

Q1 = backers.quantile(0.25)
Q3 = backers.quantile(0.75)
IQR = Q3 - Q1
filtered_backers = backers[(backers <= Q3 + 1.5 * IQR)]

num_outliers_removed = len(backers) - len(filtered_backers)

plt.hist(filtered_backers, bins=50, edgecolor='black')
plt.title('Histogram of Backers (Outliers Removed)')
plt.xlabel('Number of Backers')
plt.ylabel('Frequency')
plt.show()

print(f"Number of outliers removed: {num_outliers_removed}")
