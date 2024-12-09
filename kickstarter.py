
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional

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

# distribution of failed vs success

distribution = kickstarter_final['state'].value_counts()
print(distribution)
percentage_distribution = kickstarter_final['state'].value_counts(normalize=True) * 100
print(percentage_distribution)

kickstarter_final['state'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Distribution of Success vs Failure')
plt.ylabel('Count')
plt.show()
# %%
# state by country, currency, and category


grouped_country = kickstarter_final.groupby(['country', 'state']).size().unstack()

grouped_country.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'green'])
plt.title('Projects by Category, and Outcome (Stacked)')
plt.ylabel('Count')
plt.xlabel('Category')
plt.legend(title='Outcome')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

grouped_currency = kickstarter_final.groupby(['currency', 'state']).size().unstack()

grouped_currency.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'green'])
plt.title('Projects by Category, and Outcome (Stacked)')
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

