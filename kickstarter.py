
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional
import statsmodels.formula.api as smf

#%%

kickstarter = pd.read_csv("ks-projects-201801.csv")

print("\nReady to continue.")
# %%

kickstarter.head()
# %%
kickstarter.info()

kickstarter['state'].unique()
kickstarter['main_category'].unique()

# %%
kickstarter1 = kickstarter[kickstarter['state'].isin(['failed', 'successful'])]
print(kickstarter1)
# subset for just failed or success, reduces set to 331675 rows with 15 variables
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
# summary stats (for all countries )
# Describe continuous variables
print(kickstarter_final[['backers', 'usd_goal_real', 'usd_pledged_real', 'Duration']].describe())

# Describe categorical variables
print(kickstarter_final[['main_category', 'state']].apply(lambda x: x.describe(include='all')).T)


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
# percent failed vs percent success pie chart

succ_fail_counts = kickstarter_final['state'].value_counts()

# Data for the pie chart
labels = succ_fail_counts.index
sizes = succ_fail_counts.values

# Create pie chart
plt.pie(sizes, labels=labels, colors = ['red', 'green'], autopct='%1.1f%%', startangle=140)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Add title
plt.title('Observations: Success vs Failed')

# Show the plot
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
# Top 5 categories with the highest number of successes:
top_categories = (
    kickstarter_final[kickstarter_final['state'] == 'successful']
    .groupby('main_category')
    .size()
    .sort_values(ascending=False)
    .head(5)
)

print("Top 5 Categories with the Highest Number of Successful Projects:")
print(top_categories)

#%% 
# top 5 categories (based on percentage of successful projects in the category)

total_projects_per_category = kickstarter_final.groupby('main_category').size()
successful_projects_per_category = kickstarter_final[kickstarter_final['state'] == 'successful'].groupby('main_category').size()

# Calculate the percentage of successful projects
success_percentage = (successful_projects_per_category / total_projects_per_category) * 100

# Get the top categories with the highest percentage of successful projects
top_categories_percentage = success_percentage.sort_values(ascending=False).head(5)

print("Top 5 categories with the Highest Percentage of Successful Projects ")
print(top_categories_percentage)

#%%
# Barplot showing top 5 categories
colors = ['lightblue', 'salmon', 'lightgreen', 'wheat', 'violet']  

# Plot the top 5 categories with individual bar colors
top_categories.plot(kind='bar', color=colors, figsize=(10, 6))
plt.title('Top 5 Categories with the Most Successful Projects', fontsize= 14)
plt.ylabel('Number of Successful Projects')
plt.xlabel('Main Category')
plt.xticks(rotation=35, ha='right', fontsize = 13)

#Adding the values on to each bar
for index, value in enumerate(top_categories):
    plt.text(index, value + 100, str(value), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

#%% 
# using median goal instead of mean goal
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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
plt.title('Percentage of Successful Projects in Each Category', fontsize=14)
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
import matplotlib.pyplot as plt

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



# %%
# create a training set

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(kickstarter_final, train_size=800, random_state=42)

#%%
#fit tree to training data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_trainkickstarter = train_set.drop(columns=['state'])
y_trainkickstarter = train_set['state']

X_trainkickstarter = pd.get_dummies(X_trainkickstarter, drop_first=True)

dtree_kickstarter = DecisionTreeClassifier(max_depth = 8, criterion = 'gini', random_state = 1)

dtree_kickstarter.fit(X_trainkickstarter, y_trainkickstarter)

y_trainkickstarter_pred = dtree_kickstarter.predict(X_trainkickstarter)


training_error_rate_kickstarter = 1 - accuracy_score(y_trainkickstarter, y_trainkickstarter_pred)

print(f"Training error rate: {training_error_rate_kickstarter:.4f}")

#%%

#plot the tree

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(dtree_kickstarter, feature_names=X_trainkickstarter.columns, class_names=dtree_kickstarter.classes_, filled=True, rounded=True)
plt.title('Decision Tree for Kickstarter Campaign Outcomes')
plt.show()

n_terminal_nodes = sum(dtree_kickstarter.tree_.children_left == -1)
print(f"Number of terminal nodes (leaf nodes): {n_terminal_nodes}")

#%%

from sklearn.tree import export_text

# Generate a text summary of the tree
tree_rules = export_text(dtree_kickstarter, feature_names=X_trainkickstarter.columns.tolist())
print(tree_rules)

#%%

#predict response on the test data and produce confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

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

from sklearn.linear_model import LogisticRegression
from statsmodels.formula.api import glm
from sklearn.metrics import classification_report

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

# %%

# Creating ROC curve with AUC for logistic model
from sklearn.metrics import roc_auc_score, roc_curve

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
### logistic model (only US and using goal instead of pledged)

# subset out failed/successful
kickstarter2 = kickstarter[kickstarter['state'].isin(['failed', 'successful'])]

#subset out only US projects 
kickstarter2 = kickstarter2[kickstarter['country'].isin(['US'])]

#rename film category 
kickstarter2['main_category'] = kickstarter2['main_category'].replace({'Film & Video': 'Film_and_Video'})

# Add duration of campaign (difference between launch date and deadline)
kickstarter2['launched'] = pd.to_datetime(kickstarter2['launched']).dt.date
kickstarter2['deadline'] = pd.to_datetime(kickstarter2['deadline']).dt.date
kickstarter2['Duration'] = (pd.to_datetime(kickstarter2['deadline']) - pd.to_datetime(kickstarter2['launched'])).dt.days

# Convert categorical columns to category type
for col in ['main_category', 'currency', 'state']:
    kickstarter2[col] = kickstarter2[col].astype('category')

# Create final dataframe with selected columns
kickstarter_final_US = kickstarter2[['main_category', 'currency', 'state', 'backers', 'Duration', 'usd_goal_real']]

# Add dummy variables for categorical columns
kickstarter_final_US = pd.get_dummies(kickstarter_final_US, columns=['main_category', 'currency'], drop_first=True)

# split into features and target variables 
selected_features_us = ["backers", "usd_goal_real", "main_category_Comics", "main_category_Crafts", "main_category_Dance", "main_category_Design", "main_category_Fashion", "main_category_Film_and_Video", "main_category_Food", "main_category_Games", "main_category_Journalism", "main_category_Music", "main_category_Photography", "main_category_Publishing", "main_category_Technology", "main_category_Theater", "Duration"]

x_us = kickstarter_final_US[selected_features_us]
y_us = kickstarter_final_US['state']

# Split into training and test sets
x_us_train, x_us_test, y_us_train, y_us_test = train_test_split(x_us, y_us, test_size=0.2, random_state=42)

# Initialize the logistic regression model
logreg_us_model = LogisticRegression(max_iter=5000)

logreg_us_model.fit(x_us_train, y_us_train)

# Model Accuracy
print('Logit model accuracy (train set):', logreg_us_model.score(x_us_train, y_us_train))
print('Logit model accuracy (test set):', logreg_us_model.score(x_us_test, y_us_test))

coefficients_us = pd.DataFrame({
    'Predictors': x_us.columns,
    'Coefficient': logreg_us_model.coef_[0],
    'Odds Ratio': np.exp(logreg_us_model.coef_[0])
})
print("\nCoefficients and Odds Ratios:\n", coefficients_us)

# Predictions and Evaluation
y_pred_us = logreg_us_model.predict(x_us_test)

conf_matrix_us = confusion_matrix(y_us_test, y_pred_us)
accuracy_us = accuracy_score(y_us_test, y_pred_us)

print("\n The confusion matrix of the model is:")
print(conf_matrix_us)

print("\n The accuracy of the model is:")
print(accuracy_us)

print("\n The model's classification Report:")
print(classification_report(y_us_test, y_pred_us))

# ROC curve with AUC for logistic model
y_pred_prob_us = logreg_us_model.predict_proba(x_us_test)[:, 1]
roc_auc_us = roc_auc_score(y_us_test, y_pred_prob_us) # evaluating AUC

print(f"\n The area under the curve is found to be {roc_auc_us:.3f}.")

y_us_test_mapped = y_us_test.map({'failed': 0, 'successful': 1})

fpr, tpr, thresholds = roc_curve(y_us_test_mapped, y_pred_prob_us)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_us:.3f}")

# Shading the AUC
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.3)

# Text with AUC value inside the plot
plt.text(0.4, 0.5, f'AUC = {roc_auc_us:.3f}', fontsize=12)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# %%

kickstarter_final_US['state_binary'] = kickstarter_final_US['state'].map({'failed': 0, 'successful': 1})
kickstarter_final_US['state_binary'] = kickstarter_final_US['state_binary'].astype(int)

# Split the data into training and testing sets
train_df_stats, test_df_stats = train_test_split(kickstarter_final_US, test_size=0.2, random_state=42)

# Define the formula for logistic regression
formula = 'state_binary ~ backers + usd_goal_real + main_category_Comics + main_category_Crafts + main_category_Dance + main_category_Design + main_category_Fashion + main_category_Film_and_Video + main_category_Food + main_category_Games + main_category_Journalism + main_category_Music + main_category_Photography + main_category_Publishing + main_category_Technology + main_category_Theater + Duration'

# Fit the logistic regression model
stats_model_us = smf.logit(formula, data=train_df_stats).fit()

# Make predictions on the test data
test_df_stats['predicted'] = stats_model_us.predict(test_df_stats)
test_df_stats['predicted_class'] = (test_df_stats['predicted'] > 0.5).astype(int)

# Evaluate the model's performance
accuracy_stats_us = accuracy_score(test_df_stats['state_binary'], test_df_stats['predicted_class'])
conf_matrix_stats_us = confusion_matrix(test_df_stats['state_binary'], test_df_stats['predicted_class'])
class_report_stats_us = classification_report(test_df_stats['state_binary'], test_df_stats['predicted_class'])

print(f'Accuracy: {accuracy_stats_us}')
print('Confusion Matrix:')
print(conf_matrix_stats_us)
print('Classification Report:')
print(class_report_stats_us)

# Display the model summary
print(stats_model_us.summary())


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Extract the independent variables from the training data
independent_vars_for_vif = train_df_stats[['backers', 'usd_goal_real', 'main_category_Comics', 'main_category_Crafts', 
                                   'main_category_Dance', 'main_category_Design', 'main_category_Fashion', 
                                   'main_category_Film_and_Video', 'main_category_Food', 'main_category_Games', 
                                   'main_category_Journalism', 'main_category_Music', 'main_category_Photography', 
                                   'main_category_Publishing', 'main_category_Technology', 'main_category_Theater', 
                                   'Duration']]

# Convert boolean variables to integers
independent_vars_for_vif = independent_vars_for_vif.applymap(lambda x: int(x) if isinstance(x, bool) else x)

# Ensure all data types are numeric
independent_vars_for_vif = independent_vars_for_vif.apply(pd.to_numeric, errors='coerce')

# Create a DataFrame to store VIF values
vif_data = pd.DataFrame()
vif_data['Feature'] = independent_vars_for_vif.columns
vif_data['VIF'] = [variance_inflation_factor(independent_vars_for_vif.values, i) for i in range(len(independent_vars_for_vif.columns))]

print(vif_data)


#%% 

# take sample of dataset and do feature selection
# fit the model to full dataset
import time
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

#%%
# try tree without pledge 

# create a training set

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(kickstarter_final, train_size=800, random_state=42)

#fit tree to training data
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

X_trainkickstarter = train_set.drop(columns=['state', 'usd_pledged_real'])

y_trainkickstarter = train_set['state']

X_trainkickstarter = pd.get_dummies(X_trainkickstarter, drop_first=True)

dtree_kickstarter = DecisionTreeClassifier(max_depth = 5, criterion = 'entropy', random_state = 1)

dtree_kickstarter.fit(X_trainkickstarter, y_trainkickstarter)

y_trainkickstarter_pred = dtree_kickstarter.predict(X_trainkickstarter)


training_error_rate_kickstarter = 1 - accuracy_score(y_trainkickstarter, y_trainkickstarter_pred)

print(f"Training error rate: {training_error_rate_kickstarter:.4f}")

#plot the tree

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(dtree_kickstarter, feature_names=X_trainkickstarter.columns, class_names=dtree_kickstarter.classes_, filled=True, rounded=True)
plt.title('Decision Tree for Kickstarter Campaign Outcomes')
plt.show()

n_terminal_nodes = sum(dtree_kickstarter.tree_.children_left == -1)
print(f"Number of terminal nodes (leaf nodes): {n_terminal_nodes}")

from sklearn.tree import export_text

# Generate a text summary of the tree
tree_rules = export_text(dtree_kickstarter, feature_names=X_trainkickstarter.columns.tolist())
print(tree_rules)

#predict response on the test data and produce confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

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

maxlevels = [None, 2, 3, 5, 8]
crits = ['gini', 'entropy']
for l in maxlevels:
    for c in crits:
        dt = DecisionTreeClassifier(max_depth = l, criterion = c)
        dt.fit(X_trainkickstarter, y_trainkickstarter)
        print(l, c, dt.score(X_testkickstarter, y_testkickstarter))
        
#%%
# try tree without pledge (US only )

# create a training set

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(kickstarter_final_US, train_size=800, random_state=42)

#fit tree to training data
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

X_trainkickstarter = train_set.drop(columns=['state', 'state_binary'])

y_trainkickstarter = train_set['state']

X_trainkickstarter = pd.get_dummies(X_trainkickstarter, drop_first=True)

dtree_kickstarter = DecisionTreeClassifier(max_depth = 5, criterion = 'entropy', random_state = 1)

dtree_kickstarter.fit(X_trainkickstarter, y_trainkickstarter)

y_trainkickstarter_pred = dtree_kickstarter.predict(X_trainkickstarter)


training_error_rate_kickstarter = 1 - accuracy_score(y_trainkickstarter, y_trainkickstarter_pred)

print(f"Training error rate: {training_error_rate_kickstarter:.4f}")

#plot the tree

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(dtree_kickstarter, feature_names=X_trainkickstarter.columns, class_names=dtree_kickstarter.classes_, filled=True, rounded=True)
plt.title('Decision Tree for Kickstarter Campaign Outcomes')
plt.show()

n_terminal_nodes = sum(dtree_kickstarter.tree_.children_left == -1)
print(f"Number of terminal nodes (leaf nodes): {n_terminal_nodes}")

from sklearn.tree import export_text

# Generate a text summary of the tree
tree_rules = export_text(dtree_kickstarter, feature_names=X_trainkickstarter.columns.tolist())
print(tree_rules)

#predict response on the test data and produce confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

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

maxlevels = [None, 2, 3, 5, 8]
crits = ['gini', 'entropy']
for l in maxlevels:
    for c in crits:
        dt = DecisionTreeClassifier(max_depth = l, criterion = c)
        dt.fit(X_trainkickstarter, y_trainkickstarter)
        print(l, c, dt.score(X_testkickstarter, y_testkickstarter))
# %%

