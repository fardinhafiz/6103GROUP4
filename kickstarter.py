
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional

#%%


kickstarter = pd.read_csv("ks-projects-201801.csv")

kickstarter = pd.read_csv("/Users/rachelthomas/Desktop/GIT 6103/WORKING FILES/6103GROUP4/ks-projects-201801.csv")


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
# Barplot showing top 5 categories
colors = ['lightblue', 'salmon', 'lightgreen', 'wheat', 'violet']  

# Plot the top 5 categories with individual bar colors
top_categories.plot(kind='bar', color=colors, figsize=(10, 6))
plt.title('Top 5 Categories with the Most Successful Projects')
plt.ylabel('Number of Successful Projects')
plt.xlabel('Main Category')
plt.xticks(rotation=35, ha='right')

#Adding the values on to each bar
for index, value in enumerate(top_categories):
    plt.text(index, value + 100, str(value), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# %%

