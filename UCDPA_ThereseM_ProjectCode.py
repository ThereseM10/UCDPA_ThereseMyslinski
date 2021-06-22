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
# Using pivot-table() function to calculate the sum of characters in each alignment category by their gender.
dc_ALIGN_gender = dc_replace.pivot_table(values='name', index='ALIGN', columns='SEX', aggfunc='count', fill_value=0)
print(dc_ALIGN_gender)

# 3) Visualise
# Objective: Seaborn / Matplotlib
# Stacked Bar Chart
# Setting figure's size to 10 inches by 8.
fig, ax = plt.subplots(figsize=(10,8))
# Creating individual bars for each gender using pivot table. Male is set as the first bar as it has the most counts.
ax.bar(dc_ALIGN_gender.index, dc_ALIGN_gender["Male Characters"], label='Male')
# Stacking the female character counts on top of the male bar.
ax.bar(dc_ALIGN_gender.index, dc_ALIGN_gender["Female Characters"], bottom=dc_ALIGN_gender["Male Characters"], label='Female')
ax.bar(dc_ALIGN_gender.index, dc_ALIGN_gender["Genderless Characters"],
       bottom=dc_ALIGN_gender["Male Characters"] + dc_ALIGN_gender["Female Characters"], label='Genderless')
ax.bar(dc_ALIGN_gender.index, dc_ALIGN_gender["Transgender Characters"],
       bottom=dc_ALIGN_gender["Male Characters"] + dc_ALIGN_gender["Female Characters"] + dc_ALIGN_gender["Genderless Characters"], label='Transgender')
# Adding in main title and axis labels with adjusted font size.
ax.set_title("DC Comics: Characters' alignment and their gender", fontsize=14)
ax.set_xlabel("Alignment Categories", fontsize=12)
ax.set_ylabel("Total Number of Characters", fontsize=12)
ax.legend()
plt.show()


# Dataset 3: NBA Players
# 1) Importing Data
# Objective: Importing CSV
# Importing CSV and saving as pandas DataFrame with missing values recognised as blank spaces.
NBA_import = pd.read_csv('all_seasons.csv', na_values=' ')

# Using head() function to see first five rows of the data to ensure it imported correctly.
print(NBA_import.head())

# Using the drop() function to delete the first column as it does not contribute to the dataset.
# The axis argument is set to 1, this is to delete the whole column.
NBA_import.drop('Unnamed: 0', inplace=True, axis=1)

# 2) Analysing Data: NBA Height and Weight for Numpy objective
# Objective: Missing Values / Drop Duplicates
# Replacing missing values with NaN using the replace() function.
# The first argument is what missing values were set to be recognised as when importing and the second argument is changing them to NaN.
NBA_replace = NBA_import.replace(' ', np.nan)

# Objective: Grouping
# Using a pivot table to calculate the total number of players in each season.
# The values are sorted using the sort_values() function by the season with the highest number of players.
print(NBA_replace.pivot_table(values="player_name", index="season", aggfunc='count').sort_values('player_name', ascending=False))

# Objective: Slicing (loc / iloc)
# The index is set to be the 'season' column to make it easier to subset.
# To slice the index, it must be sorted first. This was done using the sort_index() function.
NBA_ind = NBA_replace.set_index('season').sort_index()

# The 2017-18 season is shown to have the most number of players.
# Slicing for 2017-18 season and the height and weight columns of NBA players.
# The height and weight columns are renamed using the rename() function and columns argument to allow for easier column access.
NBA_height_weight = NBA_ind.loc['2017-18', ['player_height', 'player_weight']].rename(columns={'player_height': 'height',
                                                                                                      'player_weight': 'weight'})
print(NBA_height_weight)

# Objective: Numpy
# Using the numpy array function, the height and weight columns are converted into a 2D Numpy array.
height_weight_np = np.array(NBA_height_weight[['height', 'weight']])
print(type(height_weight_np))

# Using Numpy's mean and median functions, the average height and weight of the players are calcualted and printed.
# Both mean and median were used in case the mean was affected by outliers.
mean_height = np.mean(height_weight_np[:,0])
mean_weight = np.mean(height_weight_np[:,1])
median_height = np.median(height_weight_np[:, 0])
median_weight = np.median(height_weight_np[:, 1])
# The mean and median calculations are converted to string as can only print likeable data types.
print("Mean height: " + str(mean_height) + "; Mean weight: " + str(mean_weight))
print("Median height: " + str(median_height) + "; Median weight: " + str(median_weight))

# For later data visualisation analysis, the height column was subset into another numpy array and divided by 100.
height_cm_np = np.array(NBA_height_weight['height'])
height_m_np = height_cm_np / 100

# Using the Numpy min() function to find the height of the smallest player.
print(np.min(height_m_np))

# Using Boolean operator 'and' to find how many players are smaller than the average.
print(height_m_np[np.logical_and(height_m_np > 1, height_m_np < 2)])

# Preparing height and weight data for the visualization objective.
# Slicing for the 2017-18 season rows and the players' name, points they got, and height and weight columns.
# The index was reset and then deleted to allow for easier plotting.
NBA_seventeen_season = NBA_ind.loc['2017-18', ['player_name', 'pts',
                                               'player_height',
                                               'player_weight',]].reset_index().drop(columns='season', axis=1)

# Using the pandas.Series() function, the previously calculated height in meters column replaces the height (cm) column.
NBA_seventeen_season['player_height'] = pd.Series(height_m_np)
print(NBA_seventeen_season)

# Using the replace() function and column argument, the height and weight columns are renamed to allow for ease of access.
NBA_plot = NBA_seventeen_season.rename(columns={'player_height': 'Height (m)', 'player_weight': 'Weight (kg)'})
print(NBA_plot)

# 3) Visualise
# Objective: Seaborn / Matplotlib
# Using Seaborn set() function to adjust the font size.
sns.set(font_scale=1)
# The background style is set to darkgrid to make it easier to read the plotted data.
sns.set_style("darkgrid")
# The relplot() function is used with kind argument set to scatter.
# The x and y axis are set to show height and weight, and the data argument equals the data wish to plot.
# The plot is assigned to the variable 'g' as this is common practice when using Seaborn plots.
g = sns.relplot(x='Height (m)', y='Weight (kg)', data=NBA_plot, kind='scatter')
g.fig.set_size_inches(8, 8)
# As this plot is a FacetGrid, the fig.suptitle function is used to add a title.
# The y argument sets the height of the title and the fontsize argument is to manually adjust the fontsize.
g.fig.suptitle("Height and Weight of NBA Players in the 2017-18 season", y=1)
g.set(xlabel='Height (m)', ylabel='Weight (cm)')
plt.show()

# 2) Analysing Data: NBA Players' Countries of Origin for data visualization objective.
# Objective: Looping, iterrows()
# Using For Loop on the NBA Player dataset to get the player's names who scored more than the average number of points.
NBA_most_points = NBA_seventeen_season.set_index(["player_name"])
for i, row in NBA_most_points.iterrows():
    if row['pts'] > NBA_most_points['pts'].median():
        print(i + ": " + str(row['pts']) + " points")

# Objective: Slicing
# Setting the index to country for slicing.
NBA_index = NBA_import.set_index('country').sort_index()
print(NBA_index)

# Slicing for all the rows and columns that is USA and saving to a new variable.
# Using the reset_index() function to make it easier to drop the USA column.
NBA_USA = NBA_index.loc[['USA'], :].reset_index()

# Removing USA rows and saving to a new variable.
NBA_drop = NBA_index.drop(labels='USA', axis=0).reset_index()

# Subsetting only country and season columns for data visualization objective.
NBA_other = NBA_drop[["country", "season"]]

# Objective: Grouping
# Using the groupby() and unique() functions to calculate the number of countries in each season.
# Resetted the index to allow for plotting.
NBA_plot = NBA_other.groupby('season')['country'].nunique().to_frame().reset_index()
print(NBA_plot)

# 3) Visualize
# Objective: Seaborn / Matplotlib
# Setting the context to paper as the plot will be inserted into a report document.
sns.set_context("paper")
# Using set() function to manually set the font size.
sns.set(font_scale=1)
# Using catplot to show the number of countries present in each season, excluding USA.
g = sns.catplot(x='season', y='country', data=NBA_plot, kind='bar')
# Setting the figure size.
g.fig.set_size_inches(10, 6)
# Setting FacetGrid title and axis labels
g.fig.suptitle("Number of Countries (excluding USA) in each NBA Season (1996 to 2020)", y=1)
g.set(xlabel="NBA Season", ylabel="Total Number of Countries")
# Adjusting tick rotation to make the x-axis labels easier to read.
plt.xticks(rotation=30)
plt.show()


# Dataset 4: Singapore Weather.
# 1) Importing Data
# Objective: Importing CSV
# Importing each stations' CSV file and saving as a pandas DataFrame. Missing values were recognised as blank spaces.
ang_import = pd.read_csv('angmokio.csv', sep=',', na_values=' ')
chan_import = pd.read_csv('changi.csv', sep=',', na_values=' ')

# 2) Analysing Data: Monthly Average Temperature
# Objective: Missing Values / Dropping Duplicates
# Printing the total number of missing values in each column.
print(chan_import.isna().sum())
print(ang_import.isna().sum())

# Replacing missing values with NaN.
chan_replace = chan_import.replace(' ', np.nan)
ang_replace = ang_import.replace(' ', np.nan)

# Objective: Functions
# Created a function to calculate average monthly temperature.
def monthly_temp_avg(df_replace):
    # Combining time columns into 'Date' column and converting to pandas datetime.
    df_replace['Date'] = pd.to_datetime(df_replace[["Year", "Month", "Day"]])
    # Selecting columns needed for visualization objective.
    df_subset = df_replace[["Date", "Mean Temperature (°C)"]]
    df_subset_ind = df_subset.set_index('Date').sort_index()
    # Renaming temperature column.
    df_rename = df_subset_ind.rename(columns={"Mean Temperature (°C)": "Avg Temperature (°C)"})
    # Using resample() function to calculate monthly and using mean() to calculate the average.
    df_monthly = df_rename.resample('M').mean()
    return df_monthly

# Using function to identify highest average monthly temperature and the date.
chan_monthly_temp = monthly_temp_avg(chan_replace)
print(chan_monthly_temp.sort_values('Avg Temperature (°C)', ascending=False))
ang_monthly_temp = monthly_temp_avg(ang_replace)
print(ang_monthly_temp.sort_values('Avg Temperature (°C)', ascending=False))

# Using original pandas dataframe for the daily average temperature.
chan_replace['Date'] = pd.to_datetime(chan_replace[["Year", "Month", "Day"]])
ang_replace['Date'] = pd.to_datetime(ang_replace[["Year", "Month", "Day"]])
chan_daily_temp = chan_replace[["Date", "Mean Temperature (°C)"]]
ang_daily_temp = ang_replace[["Date", "Mean Temperature (°C)"]]

# Objective: Merge DataFrames
# Using merge_ordered function to join two time series tables on the Date column.
# Setting suffixes argument to identify both stations' temperature columns.
chan_ang_merged = pd.merge_ordered(chan_daily_temp, ang_daily_temp, on='Date', suffixes=['_chan', '_ang'])
print(chan_ang_merged)

# Objective: Slicing
# Setting Date column as index.
chan_ang_merged_ind = chan_ang_merged.set_index('Date')
# Slicing the period that had the highest monthly average temperature and all columns.
chan_ang_plot = chan_ang_merged_ind.loc["2016-04-01": "2016-05-31", :]

# 3) Visualise
# Objective: Matplotlib / Seaborn
# Setting the figure size and resolution.
fig, ax = plt.subplots(figsize=(12,6), dpi=100)
# Plotting a line graph with date on the x-axis and temperature on the y.
# Setting labels for the legend and linestyle and colour.
ax.plot(chan_ang_plot.index.values, chan_ang_plot['Mean Temperature (°C)_chan'],
        label="Changi Station", linestyle="--", color='deepskyblue')
ax.plot(chan_ang_plot.index.values, chan_ang_plot['Mean Temperature (°C)_ang'],
        label="Angmokio Station", linestyle="--", color='coral')
# Setting main and axis titles with fontsize.
ax.set_title("April and May 2016 Daily Average Temperature in Angomokio and Changi", fontsize=12)
ax.set_xlabel("Time (daily)", fontsize=10)
ax.set_ylabel("Average Temperature (°C)", fontsize=10)
plt.legend()
plt.show()


# Dataset 5: World Cities Populations
# 1) Importing CSV
# Objective: Import CSV
read_csv = pd.read_csv('worldcities.csv', sep=',', na_values=[' ', '0'])
print(read_csv.head())

# Identifying and removing missing values
print(read_csv.isna().sum())
csv_dropped = read_csv.dropna()

# Subsetting city and population columns
city_pop_csv = csv_dropped[["city_ascii", "population"]]

# Renaming the columns
city_pop_renamed = city_pop_csv.rename(columns={"city_ascii": "City", "population": "Pop"})

# Objective: Dictionary
# Creating a dictionary from the city and population pandas dataframe
city_pop_dict = dict(zip(city_pop_renamed['City'], city_pop_renamed['Pop']))

# Accessing dictionary, using key 'Madrid' for the value (population)
print(city_pop_dict['Madrid'])

# Printing the key with the highest value.
highest_pop = max(city_pop_dict, key=city_pop_dict.get)
print(highest_pop, city_pop_dict[highest_pop])

# Printing the key with the lowest value.
lowest_pop = min(city_pop_dict, key=city_pop_dict.get)
print(lowest_pop, city_pop_dict[lowest_pop])
