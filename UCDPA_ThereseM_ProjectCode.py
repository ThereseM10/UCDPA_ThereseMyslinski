# Certificate in Introductory Data Analytics
# Real World Datasets
# Dataset 1: Greenhouse Gas Emissions (1999-2008) dataset.
# Dataset 2: Comic Characters.
# Dataset 3: NBA Players.
# Dataset 4: Singapore Weather.
# Dataset 5: World Cities Populations.

# Importing packages
# Importing numpy to use for statistical calculations and pandas for cleaning and manipulating data.
import numpy as np
import pandas as pd

# Importing matplotlib and seaborn for data visualization.
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset 1: Greenhouse Gas Emissions (1999- 2008)
# 1) Importing Data
# Objective: Importing API
# Importing the requests packaged used for packaging and sending a request to the specified url link.
import requests

# Saving dataset url to a variable.
url_api = 'https://ws.cso.ie/public/api.jsonrpc?data=%7B%22jsonrpc%22:%222.0%22,%22method%22:%22PxStat.Data.Cube_API.ReadDataset%22,%22params%22:%7B%22class%22:%22query%22,%22id%22:%5B%5D,%22dimension%22:%7B%7D,%22extension%22:%7B%22pivot%22:null,%22codes%22:false,%22language%22:%7B%22code%22:%22en%22%7D,%22format%22:%7B%22type%22:%22JSON-stat%22,%22version%22:%222.0%22%7D,%22matrix%22:%22EAA05%22%7D,%22version%22:%222.0%22%7D%7D'

# Using get() function to package and send a request, and capture the file into a variable.
request = requests.get(url_api)

# Used the JSON decoder, that returned a dictionary, to a variable.
GHG_data = request.json()

# Using a for loop, using key and value pairs (as data is in a dictionary), to review what was captured from the API.
for key, value in GHG_data.items():
    print(key + ': ', value)


# Dataset 2: Comic Characters.
# 1) Importing Data
# Objective: Importing CSV
# Importing CSV file and saving as a Pandas DataFrame.
# Missing values are recognised as blank spaces.
dc_import = pd.read_csv('dc-wikia-data.csv', sep=',', na_values=' ')

# Summarising information in the imported file.
print(dc_import.head())
print(dc_import.shape)
print(dc_import.columns)

# Objective: Missing Values / Drop Duplicates
# Calculating the total number of missing values in each column and replacing with 'NaN'
print(dc_import.isna().sum())
dc_replace = dc_import.replace(' ', np.nan)

# 2) Analysing Data
# Objective: Grouping
# Using pivot-table() function to calculate the sum of characters in each gender category and their alignment.
dc_gender_ALIGN = pd.pivot_table(dc_replace, index=['SEX', 'ALIGN'], aggfunc={'name': 'count'})
print(dc_gender_ALIGN)

# Objective: Sorting
# Sorting values to see gender and alignment with the least number of characters.
print(dc_gender_ALIGN.sort_values(('name'), ascending=True))

# 3) Visualise
# Objective: Seaborn / Matplotlib
# Setting the figure style.
sns.set_style("ticks")
# Setting the scale.
sns.set_context("paper")
g = sns.countplot(x='SEX', hue="ALIGN", data=dc_replace)
g.axes.set_title("DC Comics: Characters' gender and their alignment", fontsize=12)
g.set_xlabel("Gender Category", fontsize=12)
g.set_ylabel("Total Number of Characters", fontsize=12)
plt.legend(bbox_to_anchor=(1, 1), loc=1)
plt.show()

# Objective: Merging DataFrames
# Importing Marvel Comics csv file and saving as a pandas dataframe.
# Missing values recognised as blank.
marvel_import = pd.read_csv('marvel-wikia-data.csv', sep=',', na_values=' ')

# To check dataset imported correctly.
print(marvel_import.head())

# Checking column names for merging objective.
print(marvel_import.columns)

# Replacing missing values with NaN.
marvel_replace = marvel_import.replace(' ', np.nan)

# Using pivot_table() function on both DC and Marvel to see how many characters firsted appeared, their gender and alignment.
dc_pivot = dc_replace.pivot_table(values="name", index=["YEAR", "SEX"], columns="ALIGN", aggfunc='count',fill_value=0)
marvel_pivot = marvel_replace.pivot_table(values="name", index=["Year", "SEX"], columns="ALIGN", aggfunc='count',fill_value=0)

# Merging Marvel onto DC (left table) as DC has the longest year column.
# Using right_on parameter to use Year and SEX columns and using suffixes for similar columns in both tables.
dc_marvel_merge = dc_pivot.merge(marvel_pivot, how='left', left_index=True,
                                 right_on=['Year', 'SEX'], suffixes=['_dc', '_marvel'])
print(dc_marvel_merge)

