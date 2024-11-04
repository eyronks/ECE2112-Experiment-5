# [ECE2112] Exploratory Data Analysis (EDA) on Spotify 2023 Data Set

This repository presents an exploratory data analysis (EDA) of the Most Streamed Spotify Songs 2023 dataset from Kaggle. The goal of this analysis is to examine streaming data and musical attributes of popular tracks to extract meaningful insights into the factors that contribute to a song's popularity.

<br>

## Analysis Overview 

This EDA is structured around the following key questions:

  1. Overview of Dataset
     * Identify the number of rows and columns, data types, and any missing values in the dataset.
    
  2. Basic Descriptive Statistics
     * Compute mean, median, and standard deviation of the streams column.
     * Analyze distributions for released_year and artist_count to detect trends or outliers.
    
  3. Top Performers
     * Identify the top 5 most-streamed tracks and the top 5 most frequent artists.
       
  4. Temporal Trends
     * Explore the trends in track releases over time by year and month, identifying peak release months.
    
  5. Genre and Music Characteristics
     * Examine correlations between streams and musical attributes (e.g., BPM, danceability, energy) to determine which features most influence popularity.
     * Analyze correlations among attributes like danceability_% and energy_% as well as valence_% and acousticness_%.

  6. Platform Popularity
     * Compare the number of tracks featured across Spotify and Apple playlists and charts, identifying platform preferences for popular tracks.

  7. Advanced Analysis
     * Assess patterns in streams by musical key or mode, and analyze which artists or genres frequently appear across playlists and charts.

<br>

## Author's Approach

**Import the necessary libraries for the script**
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

> **Function:** These libraries are essential to solve the problem because they provide the necessary tools for data manipulation, analysis, and visualization, as well as enhancing the aesthetics of the printed data frames.

<br>

**Read a comma-separated values (CSV) file into a DataFrame**
```python
# Read the CSV file 'spotify2023.csv' into a DataFrame
# Specify encoding as 'latin1' to handle special characters correctly
df = pd.read_csv('spotify2023.csv', encoding='latin1')
```
> **Function:** The function reads 'spotify2023.csv' with 'latin1' encoding to handle special characters, ensuring accurate data import.

<br>

## Overview of the Data Set
**Determine the number of rows and columns of the data set**
```python
# Display the first five rows of the DataFrame 'df'
# This helps to quickly inspect the data and verify its structure and contents
df.head()
```
> **Function:** By default, it shows the first five rows, which provides a quick overview of the data structure, including the column names and the types of data contained within.

![image](https://github.com/user-attachments/assets/dc0b177d-e07a-4346-a8df-74f335c9f589)

```python
# Display a summary of the DataFrame 'df', including the number of entries, non-null values, and data types
df.info()
```

![image](https://github.com/user-attachments/assets/aee7044c-02e2-457a-b8c0-1d7b1a318d40)

<br>

**What are the data types of each column?**
```python
# Convert the 'in_deezer_playlists' column to numeric type, coercing errors to NaN
df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'], errors='coerce')

# Convert the 'streams' column to numeric type, coercing errors to NaN
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# Convert the 'in_shazam_charts' column to numeric type, coercing errors to NaN
df['in_shazam_charts'] = pd.to_numeric(df['in_shazam_charts'], errors='coerce')

# Print the data types of each column in the DataFrame to verify conversions
print(df.dtypes)
```
> **Function:** For each column—in_deezer_playlists, streams, and in_shazam_charts—the function attempts to convert the values to numeric types, replacing any non-convertible values with NaN (Not a Number) due to the errors='coerce' argument.

![image](https://github.com/user-attachments/assets/f1dfd751-45d0-4d75-b332-3378ec82bb21)

<br>

**Are there any missing and duplicated values?**
```python
# Count the number of missing values (NaN) in each column of the DataFrame 'df'
missing_values_per_column = df.isna().sum()

# Print the total number of missing values in the entire DataFrame
# The first sum() counts NaNs per column, and the second sum() totals them
print(f"Missing values: {missing_values_per_column.sum()}")
```
> **Function:** It counts the number of missing values in each column and then calculates the total number of missing values in the entire DataFrame.

<br>

```python
# Check for duplicate rows in the DataFrame 'df' and count them
duplicate_count = df.duplicated().sum()

# Print the total number of duplicated rows in a formatted string
print(f"Duplicated values: {duplicate_count}")
```
> **Function:** The code checks for duplicate rows in the DataFrame df. The df.duplicated() function returns a boolean Series indicating whether each row is a duplicate of a previous row. The sum() method counts the number of duplicate rows. Finally, it prints the total count of duplicated values in a formatted string.

![image](https://github.com/user-attachments/assets/c8e9598d-1c18-4bb4-9191-3d6328b25d77)

<br>

## Basic Descriptive Statistics
**What are the mean, median, and standard deviation of the streams column?**
```python
# Present descriptive statistics for the 'streams' column
df['streams'].describe()
```
> **Function:** The code generates descriptive statistics for the 'streams' column in the DataFrame df. Prior to this operation, data cleaning was performed on the 'streams' column to handle missing and NaN values, ensuring that the descriptive statistics accurately reflect the available data.

![image](https://github.com/user-attachments/assets/058cc202-3cc3-4dc3-b34a-b58fe77f2289)

<br>

**What is the distribution of released_year and artist_count? Are there any noticeable trends or outliers?**
```python
# Set the style for Seaborn
sns.set(style='whitegrid', palette='pastel')

# Set up the figure and axes for subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trend of released_year
released_year_counts = df['released_year'].value_counts().sort_index().reset_index()
released_year_counts.columns = ['released_year', 'count']

sns.lineplot(data=released_year_counts, x='released_year', y='count', ax=axes[0, 0], marker='o', color='lightblue')
axes[0, 0].set_title('Trend of Released Year')
axes[0, 0].set_xlabel('Released Year')
axes[0, 0].set_ylabel('Number of Releases')

# Box plot for outliers in released_year
sns.boxplot(x=df['released_year'], ax=axes[0, 1], orient='h', color='lightcoral')
axes[0, 1].set_title('Box Plot of Released Year')
axes[0, 1].set_xlabel('Released Year')

# Trend of artist_count
artist_count_counts = df['artist_count'].value_counts().sort_index().reset_index()
artist_count_counts.columns = ['artist_count', 'frequency']

sns.lineplot(data=artist_count_counts, x='artist_count', y='frequency', ax=axes[1, 0], marker='o', color='lightgreen')
axes[1, 0].set_title('Trend of Artist Count')
axes[1, 0].set_xlabel('Artist Count')
axes[1, 0].set_ylabel('Frequency')

# Box plot for outliers in artist_count
sns.boxplot(x=df['artist_count'], ax=axes[1, 1], orient='h', color='lightyellow')
axes[1, 1].set_title('Box Plot of Artist Count')
axes[1, 1].set_xlabel('Artist Count')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
```
> **Function:** This script uses Seaborn to visualize data from a DataFrame df. It calculates and plots the number of releases per year with a line plot to illustrate trends, and displays a box plot to identify outliers in the 'released_year' data. It also plots the frequency of different artist counts using a line plot and highlights outliers in the 'artist_count' with another box plot. Finally, the layout is adjusted for better spacing, and the plots are displayed, effectively visualizing trends and distributions for both 'released_year' and 'artist_count'.

![image](https://github.com/user-attachments/assets/1bd8d0e0-a75e-4c55-80cc-82e710908fcc)

<br>

## Top Performers
**Which track has the highest number of streams?**
```python
# Create a bar chart for the top 5 most streamed tracks
top_streamed_tracks = df[['track_name', 'streams']].sort_values(by='streams', ascending=False).head(5)

# Set up the figure
plt.figure(figsize=(10, 6))

# Create a horizontal bar chart
plt.barh(top_streamed_tracks['track_name'], top_streamed_tracks['streams'], color='skyblue')

# Set labels and title
plt.xlabel('Number of Streams')
plt.title('Top 5 Most Streamed Tracks')

# Invert y-axis to display the highest streams on top
plt.gca().invert_yaxis()

# Show the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
```
> **Function:** This code creates a horizontal bar chart to visualize the top 5 most streamed tracks from a DataFrame df. It selects and sorts the 'track_name' and 'streams' columns, retains the top 5 entries, and then plots the data. The chart includes x-axis labels and a title, inverts the y-axis to display the highest streams at the top, and adjusts the layout before displaying the plot.

![image](https://github.com/user-attachments/assets/8afa42ba-d30c-4f07-bbb6-81b60e8ab3fe)

<br>

**Who are the top 5 most frequent artists based on the number of tracks in the dataset?**
```python
# Create a horizontal bar chart for the top 5 most frequent artists
top_artists = df['artist(s)_name'].value_counts().head(5)

# Set up the figure
plt.figure(figsize=(10, 6))

# Create a horizontal bar chart
plt.barh(top_artists.index, top_artists.values, color='skyblue')

# Set labels and title
plt.xlabel('Number of Tracks')
plt.title('Top 5 Most Frequent Artists')

# Invert y-axis to display the most frequent artists on top
plt.gca().invert_yaxis()

# Show the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
```
> **Function:** This code creates a horizontal bar chart to visualize the top 5 most frequent artists from a DataFrame df. It counts the occurrences of each artist in the 'artist(s)_name' column, selects the top 5, and then plots the data. The chart is configured with specified dimensions, includes x-axis labels and a title, inverts the y-axis to display the most frequent artists at the top, and adjusts the layout for better spacing before displaying the plot.

![image](https://github.com/user-attachments/assets/448b4a72-1299-4def-a7e8-4ec6b9ecc91b)

<br>

## Temporal Trends
**Analyze the trends in the number of tracks released over time. Plot the number of tracks released per year.**
```python
# Calculate total streams per year
streams_per_year = df.groupby('released_year')['streams'].sum()

# Identify the top years with the most streams
top_years = streams_per_year.nlargest(5).index  # Get the top 5 years

# Prepare data for plotting
tracks_per_year = df['released_year'].value_counts().sort_index()

# Filter tracks_per_year to only include top years
filtered_tracks_per_year = tracks_per_year[tracks_per_year.index.isin(top_years)]

# Plot the number of tracks released in top streamed years
plt.figure(figsize=(10, 6))
plt.plot(filtered_tracks_per_year.index, filtered_tracks_per_year.values, marker='o', color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Tracks Released')
plt.title('Number of Tracks Released in Years with Most Streams')
plt.xticks(filtered_tracks_per_year.index)  # Show only top years on the x-axis
plt.grid()
plt.show()
```
> **Function:** The script analyzes trends in the number of music tracks released over time, specifically focusing on the years with the highest total streams. The purpose is to visualize the relationship between the number of tracks released and the years that garnered the most streams.

![image](https://github.com/user-attachments/assets/3646b243-84f6-4097-882d-8dc5c8f569b7)

<br>

**Does the number of tracks released per month follow any noticeable patterns? Which month sees the most releases?**
```python
# Count the number of tracks released per month
tracks_per_month = df['released_month'].value_counts().sort_index()

# Identify the month with the most releases
most_releases_month = tracks_per_month.idxmax()
most_releases_count = tracks_per_month.max()

# Plot the number of tracks released per month
plt.figure(figsize=(10, 6))
plt.bar(tracks_per_month.index, tracks_per_month.values, color='skyblue')
plt.xlabel('Month')
plt.ylabel('Number of Tracks Released')
plt.title('Number of Tracks Released Per Month')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

# Create a mapping of month numbers to month names
month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Get the name of the month with the most releases
most_releases_month_name = month_names[most_releases_month]

print(f"\nThe month with the most releases is: {most_releases_month_name} with {most_releases_count} releases.")
```
> **Function:** This function analyzes monthly music track releases to identify patterns and determine which month has the highest number of releases. It counts the tracks for each month using the 'released_month' column, identifies the month with the most releases, and creates a bar chart to visualize the data.

![image](https://github.com/user-attachments/assets/9f7e7b06-d683-48d7-b0fe-d5ba85859b58)

<br>

## Genre and Music Characteristics
**Examine the correlation between streams and musical attributes like bpm, danceability_%, and energy_%.**
```python
# Select relevant columns
attributes = ['streams', 'bpm', 'danceability_%', 'energy_%']
data = df[attributes]

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Print the correlation coefficients
print("Correlation Coefficients:")
print(correlation_matrix)

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create a heatmap to visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='mako', fmt=".2f", square=True, cbar_kws={"shrink": .8})

plt.title('Correlation Matrix between Streams and Musical Attributes')
plt.show()
```
> **Function:** The purpose of this code is to analyze the correlation between music streams and attributes like bpm, danceability percentage, and energy percentage. It calculates the correlation matrix and visualizes it with a heatmap, allowing for easy identification of which attributes most influence streams. This approach effectively combines numerical analysis and visual representation to provide clear insights into the relationships between the variables.

![image](https://github.com/user-attachments/assets/135b6c0a-4f9b-4b5a-8a44-9ca5e1fd3eb0)
![image](https://github.com/user-attachments/assets/42ca1622-be2d-4fe5-9dec-0b1e3765702e)

<br>

**Which attributes seem to influence streams the most?**
```python
# Set up the figure for scatter plots
plt.figure(figsize=(15, 10))

# Scatter plot for Streams vs BPM
plt.subplot(2, 2, 1)  # Create a 2x2 grid, first subplot
plt.scatter(df['bpm'], df['streams'], alpha=0.6, color='#66b3ff')  # Pastel blue
plt.title('Streams vs BPM')
plt.xlabel('BPM')
plt.ylabel('Streams')
plt.grid()

# Scatter plot for Streams vs Danceability
plt.subplot(2, 2, 2)  # Second subplot
plt.scatter(df['danceability_%'], df['streams'], alpha=0.6, color='#99ff99')  # Pastel green
plt.title('Streams vs Danceability %')
plt.xlabel('Danceability %')
plt.ylabel('Streams')
plt.grid()

# Scatter plot for Streams vs Energy
plt.subplot(2, 2, 3)  # Third subplot
plt.scatter(df['energy_%'], df['streams'], alpha=0.6, color='#ff9999')  # Pastel red
plt.title('Streams vs Energy %')
plt.xlabel('Energy %')
plt.ylabel('Streams')
plt.grid()

# Adjust layout for better spacing
plt.tight_layout()

# Display the plots
plt.show()
```
> **Function:** The purpose of this code is to create scatter plots that visualize the relationship between music streams and attributes like bpm, danceability percentage, and energy percentage. This visual approach allows for quick identification of how each attribute influences streams. It effectively highlights potential correlations, making it easier to determine which attributes have the strongest impact.

![image](https://github.com/user-attachments/assets/0e8bcb4f-4eb8-4b2d-a167-042b3b751df1)

<br>

**Is there a correlation between danceability_% and energy_%? How about valence_% and acousticness_%?**
```python
# Select relevant columns for analysis
features = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%']
data = df[features]

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Heatmap for Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='mako', fmt=".2f", square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()
```
> **Function:** The purpose of this code is to analyze the correlation between music attributes—danceability, energy, valence, and acousticness—by calculating and visualizing their correlation matrix. This approach effectively highlights the strength of relationships between these attributes, providing clear insights into which factors may influence music streams. The heatmap makes it easy to identify significant correlations at a glance.

![image](https://github.com/user-attachments/assets/2105f390-968a-4bce-bf3f-e12d42bb931a)

<br>

## Platform Popularity
**How do the numbers of tracks in spotify_playlists, deezer_playlists, and apple_playlists compare? Which platform seems to favor the most popular tracks?**
```python
# Rename the columns for clarity
df.rename(columns={
    'in_spotify_playlists': 'Spotify',
    'in_apple_playlists': 'Apple Music',
    'in_deezer_playlists': 'Deezer'
}, inplace=True)

# Define the new columns for the platforms
platforms = ['Spotify', 'Apple Music', 'Deezer']

# Count the number of tracks in each platform's playlists
track_counts = df[platforms].sum()

# Print the track counts for each platform
print(track_counts)

# Create a bar graph to visualize the number of tracks in each platform's playlists
plt.figure(figsize=(8, 5))
# Use pastel colors for the bars
plt.bar(track_counts.index, track_counts.values, color=['#ff9999', '#66b3ff', '#99ff99'])  # Pastel colors
plt.title("Number of Tracks in Each Platform's Playlists")
plt.xlabel('Platform')
plt.ylabel('Number of Tracks')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
```
> **Function:** The purpose of this code is to compare the number of tracks in playlists across Spotify, Apple Music, and Deezer. It renames columns for clarity, counts the total tracks for each platform, and visualizes the results with a bar graph. This approach effectively highlights which platform features the most tracks, allowing for quick insights into platform preferences for popular music.

![image](https://github.com/user-attachments/assets/b6e93e7f-2000-4d26-8ea6-3dd9f9b94125)


##  Advanced Analysis
**Based on the streams data, can you identify any patterns among tracks with the same key or mode (Major vs. Minor)?**
```python
# Set up the figure and axes
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# Total streams per key
total_streams = df.groupby('key')['streams'].sum().reset_index().sort_values('streams', ascending=False)
sns.barplot(data=total_streams, x='key', y='streams', ax=ax[0], palette='magma', hue='key', legend=False)

# Top 100 tracks by streams
top_tracks = df.sort_values('streams', ascending=False).head(100)
top_streams = top_tracks.groupby('key')['streams'].sum().reset_index().sort_values('streams', ascending=False)
sns.pointplot(data=top_streams, x='key', y='streams', color='orange', label='Top 100 Tracks', ax=ax[0])
ax[0].legend()
ax[0].set_title('Total Streams per Key', fontsize=16)

# Stream distribution per key
sns.boxplot(data=df, x='key', y='streams', ax=ax[1], palette='pastel', hue='key', legend=False)
ax[1].set_title('Stream Distribution per Key', fontsize=16)

# Track count per key
sns.countplot(data=df, y='key', order=df['key'].value_counts().index, palette='Blues', ax=ax[2], hue='key', legend=False)
ax[2].set_title('Track Count per Key', fontsize=16)

# Overall title
plt.suptitle('Key Analysis of Streaming Data', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
plt.show()
```
> **Function:** The purpose of this code is to analyze streaming data by musical keys through three visualizations. It creates a bar plot to show total streams per key, highlighting the top 100 tracks, a box plot to illustrate the distribution of streams for each key, and a count plot to display the number of tracks in each key. This approach effectively provides a comprehensive view of key performance in streaming, allowing for quick identification of trends and patterns in the data.

![image](https://github.com/user-attachments/assets/45814502-e447-47be-9d6e-aed11cebb2b4)

<br>

```python
# Set up the figure and axes
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# Total streams per mode
x = df.groupby('mode')['streams'].sum().reset_index().sort_values('streams', ascending=False)
sns.barplot(data=x, y='streams', x='mode', ax=ax[0], hue='mode', palette='magma', legend=False)

# Top 100 tracks by streams per mode
y = df.sort_values('streams', ascending=False).head(100).groupby('mode')['streams'].sum().reset_index().sort_values('streams', ascending=False)
sns.pointplot(data=y, y='streams', x='mode', color='blue', label='Top 100 Tracks', ax=ax[0])
ax[0].legend()
ax[0].title.set_text('Total Streams per Mode')

# Stream distribution per mode
sns.boxplot(data=df, x='mode', y='streams', ax=ax[1], hue='mode', palette='pastel', legend=False)
ax[1].title.set_text("Stream Distribution per Mode")

# Track count per mode
sns.countplot(data=df, y='mode', order=df['mode'].value_counts().index, ax=ax[2], hue='mode', palette='Blues', legend=False)
ax[2].title.set_text("Track Count per Mode")

# Overall title
plt.suptitle('Mode Analysis for Streaming Data', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
plt.show()
```
> **Function:** This code analyzes streaming data by musical modes through three visualizations: a bar plot for total streams per mode, a box plot for stream distribution, and a count plot for the number of tracks per mode. This approach effectively highlights trends and insights into the performance and characteristics of different musical modes, providing a comprehensive overview of their streaming success.

![image](https://github.com/user-attachments/assets/a94afe43-56ac-409d-9658-e1e5169a247f)

<br>

**Do certain genres or artists consistently appear in more playlists or charts? Perform an analysis to compare the most frequently appearing artists in playlists or charts.**
```python
# Count the occurrences of each artist in Spotify, Apple, and Deezer playlists
spotify_artist_counts = df['artist(s)_name'].value_counts().reset_index()
spotify_artist_counts.columns = ['artist', 'spotify_count']

apple_artist_counts = df['artist(s)_name'].value_counts().reset_index()
apple_artist_counts.columns = ['artist', 'apple_count']

deezer_artist_counts = df['artist(s)_name'].value_counts().reset_index()
deezer_artist_counts.columns = ['artist', 'deezer_count']

# Merge the counts into a single DataFrame
artist_counts = spotify_artist_counts.merge(apple_artist_counts, on='artist', how='outer')
artist_counts = artist_counts.merge(deezer_artist_counts, on='artist', how='outer')

# Fill NaN values with 0
artist_counts.fillna(0, inplace=True)

# Set up the figure and axes
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# Bar plot for the most frequently appearing artists in Spotify playlists
sns.barplot(data=artist_counts.nlargest(10, 'spotify_count'), x='spotify_count', y='artist', ax=ax[0], hue='artist', palette='magma', legend=False)
ax[0].title.set_text('Top 10 Artists in Spotify Playlists')
ax[0].set_xlabel('Number of Playlists')
ax[0].set_ylabel('Artist')

# Bar plot for the most frequently appearing artists in Apple playlists
sns.barplot(data=artist_counts.nlargest(10, 'apple_count'), x='apple_count', y='artist', ax=ax[1], hue='artist', palette='Blues', legend=False)
ax[1].title.set_text('Top 10 Artists in Apple Playlists')
ax[1].set_xlabel('Number of Playlists')
ax[1].set_ylabel('Artist')

# Bar plot for the most frequently appearing artists in Deezer playlists
sns.barplot(data=artist_counts.nlargest(10, 'deezer_count'), x='deezer_count', y='artist', ax=ax[2], hue='artist', palette='pastel', legend=False)
ax[2].title.set_text('Top 10 Artists in Deezer Playlists')
ax[2].set_xlabel('Number of Playlists')
ax[2].set_ylabel('Artist')

# Overall title
plt.suptitle('Artist Analysis in Playlists Across Platforms', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
plt.show()
```
> **Function:** This code analyzes the frequency of artists in playlists across Spotify, Apple Music, and Deezer. It counts occurrences of each artist, merges the data, and fills missing values with zero. Three bar plots visualize the top 10 artists for each platform, allowing for direct comparison of artist popularity. This approach effectively highlights trends in listener preferences across different streaming services.

![image](https://github.com/user-attachments/assets/f76e3fe8-8f66-458f-b2fa-0c5750edba5d)

<br>

## Data Discussion and Insights
**Overview of Dataset**
> How many rows and columns does the dataset contain?
  * The data set contains 954 rows and 22 columns in total.
<br>

> What are the data types of each column? Are there any missing values?
  * To swiftly analyze the dataset, a data cleaning process was performed. The dataset contains various data types, including object, int64, and float64. There are 232 missing values, and there are no duplicated entries.
<br>

**Basic Descriptive Statistics**
> What are the mean, median, and standard deviation of the streams column?
  * The dataset on streams consists of 952 entries, with a mean value of approximately 514,137,400 and a standard deviation of about 566,856,900. The minimum number of streams recorded is 2,762, while the maximum reaches 3,703,895,000. The 25th percentile is around 141,636,200, the median (50th percentile) is approximately 290,530,900, and the 75th percentile is about 673,869,000.
<br>

> What is the distribution of released_year and artist_count? Are there any noticeable trends or outliers?
  * The plots show the distribution of released year and artist count in a dataset. The trend of released year shows a spike in releases around 2020, with a few releases earlier in the 1900s. The box plot of released year shows that most releases are in the late 1900s, while there are some outliers in earlier years. The trend of artist count shows that the majority of songs have 1-2 artists. The box plot of artist count shows that there are some outliers with more than 4 artists, but most songs have 1-2 artists.
<br>

**Top Performers**
> Which track has the highest number of streams? Display the top 5 most streamed tracks.
  * The most streamed track is "Blinding Lights" with 3.6 billion streams. The least streamed track on this list is "Sunflower - Spider-Man: Into the Spider-Verse", which has been streamed 2.8 billion times. "Shape of You" was streamed 3.5 billion times, "Someone You Loved" was streamed 2.9 billion times, and "Dance Monkey" was streamed 2.8 billion times.
<br>

> Who are the top 5 most frequent artists based on the number of tracks in the dataset?
  * The bar chart shows the top 5 most frequent artists, with Taylor Swift having the most tracks at 33. The Weeknd is second with 22 tracks, followed by Bad Bunny with 19 tracks. SZA has 19 tracks as well, and Harry Styles has 17 tracks.
<br>

**Temporal Trends**
> Analyze the trends in the number of tracks released over time. Plot the number of tracks released per year.
  * The number of tracks released has been increasing over time, although the rate of increase is not consistent. The number of tracks released in 2022 was significantly higher than in any other year. It seems likely that the number of tracks released in 2023 will continue this trend, but it's impossible to be sure without further data.
<br>

> Does the number of tracks released per month follow any noticeable patterns? Which month sees the most releases?
  * The number of tracks released per month seems to generally increase over the year, reaching a peak in November. January is the month with the most releases.
<br>

**Genre and Music Characteristics**
> Examine the correlation between streams and musical attributes like bpm, danceability_%, and energy_%. Which attributes seem to influence streams the most?
  * The correlation matrix shows that there is a very weak relationship between streams and any of the musical attributes. The highest correlation is between streams and energy_%, with a value of 0.20. This indicates a very weak positive relationship, meaning that songs with higher energy levels might be slightly more likely to have more streams. However, the overall correlations are so low that it's safe to say that these attributes do not have a significant impact on streams.
<br>

> Is there a correlation between danceability_% and energy_%? How about valence_% and acousticness_%?
  * The correlation between danceability_% and energy_% is 0.198095. This indicates a weak positive correlation, meaning that as danceability increases, energy tends to slightly increase as well.
    
  * The correlation between valence_% and acousticness_% is -0.081907. This indicates a very weak negative correlation, meaning that as valence increases, acousticness tends to slightly decrease.

<br>

**Platform Popularity**
> How do the numbers of tracks in spotify_playlists, deezer_playlists, and apple_playlists compare? Which platform seems to favor the most popular tracks?
  * The number of tracks in Spotify playlists is far greater than the number of tracks in either Deezer or Apple Music playlists. This suggests that Spotify favors more popular tracks.
<br>

**Advanced Analysis**
> Based on the streams data, can you identify any patterns among tracks with the same key or mode (Major vs. Minor)?
  * The most streamed key is C#, followed by G. The most streamed mode is Major. This suggests that major keys are generally more popular than minor keys. However, the box plots show a wider distribution of streams for minor keys, suggesting that there are some very popular minor-key tracks, even if they are less numerous overall.
<br>

> Do certain genres or artists consistently appear in more playlists or charts? Perform an analysis to compare the most frequently appearing artists in playlists or charts.
  * The analysis shows that Taylor Swift, The Weeknd, and Bad Bunny appear among the top 10 artists in all three platforms: Spotify, Apple Music, and Deezer. This indicates that these artists have a broad appeal across different streaming services and are consistently popular across different audiences.

  * While SZA and Harry Styles are present in all three lists, their rankings vary slightly. This suggests that their popularity may be more platform-specific, with one platform potentially having a larger following for them than the others.

  * Kendrick Lamar and Morgan Wallen are consistently present in the top 10, though their positions vary across platforms. This suggests they appeal to specific genres and audiences who may prefer one streaming service over another.

  * Ed Sheeran, BTS, and Drake, 21 Savage are present in the top 10 in Spotify and Apple Music but not in Deezer. This indicates that these artists might have a more limited appeal within the Deezer audience, potentially due to their genre or style of music.

  * The analysis shows that the top 10 most frequently appearing artists across platforms are a combination of pop, hip hop, and R&B artists, highlighting the broad appeal of these genres within the streaming music world. However, differences in ranking on various platforms suggest that specific artists might resonate more strongly with audiences on particular services, possibly due to platform-specific algorithms or listener demographics.

<br>

## Conclusion

> How was the problem approached by the Author?

The author's analysis of the Spotify 2023 data is thorough and methodical, utilizing a range of exploratory data analysis (EDA) techniques. The analysis begins with a detailed examination of the dataset, including its size, column types, and identifying any missing or duplicate entries. This foundational step lays the groundwork for subsequent analysis.


Descriptive statistics, such as mean, median, and standard deviation of the streams column, are calculated to understand the central tendency and variability of the data. To explore trends and patterns, the author analyzes temporal trends by examining the number of tracks released over time, both annually and monthly, which helps to understand potential seasonality and growth patterns in the music industry.


The analysis also explores the relationships between musical attributes (like bpm, danceability, energy, valence, and acousticness) and the number of streams. Correlation matrices and scatter plots are utilized to visualize these relationships and identify potential factors influencing popularity. To understand platform preferences, the author compares the number of tracks featured on different platforms like Spotify, Apple Music, and Deezer, providing insights into how popular tracks are distributed across streaming services.


Furthermore, the analysis delves into advanced features like musical key and mode, identifying patterns in streaming data for different keys and modes. The author also analyzes the frequency of artists appearing in playlists and charts, providing insights into artist popularity across different platforms.


In summary, the author leverages a combination of data visualization, statistical analysis, and comparison techniques to extract meaningful insights into the factors influencing music popularity on Spotify. This EDA provides a comprehensive understanding of the trends, attributes, and platform preferences that contribute to a song's success.


##

**Author:** Aaron Chastine V. Villajin

**Submitted to:** Engr. Ma. Madecheen S. Pangaliman, M.Sc.
