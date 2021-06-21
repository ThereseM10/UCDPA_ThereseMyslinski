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

