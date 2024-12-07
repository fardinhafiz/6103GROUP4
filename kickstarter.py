
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional

#%%

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

kickstarter1['Date'] = kickstarter1['launched'].dt.date


print(kickstarter1.head())

kickstarter1['Date'] = pd.to_datetime(kickstarter1['Date'])
kickstarter1['deadline'] = pd.to_datetime(kickstarter1['deadline'])

# Calculate duration
kickstarter1['Duration'] = kickstarter1['deadline'] - kickstarter1['Date']

print(kickstarter1.head())

kickstarter1.info()
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

