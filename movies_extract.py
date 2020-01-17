# %%
# Import dependencies
import json 
import pandas as pd 
import numpy as np 


# %%
# set file directory for data import
file_dir = 'C:/Users/sable/Documents/Professional Projects/Berkeley Data Analytics Bootcamp/Analysis Projects/Movies-ETL/'


# %%
# Load movies json file
with open(f'{file_dir}wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)

len(wiki_movies_raw)

# %%
# First 5 records
wiki_movies_raw[:5]

# %%
# Last 5 records
wiki_movies_raw[-5:]

# %%
# middle records
wiki_movies_raw[3600:3605]

# %%
# Load Kaggle data
kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(f'{file_dir}ratings.csv')

# %%
# Inspect data
ratings.sample(5)

# %%
kaggle_metadata.count()

# %%
kaggle_metadata.dtypes

# %%
# Transform wiki raw file into df
wiki_movies_df = pd.DataFrame(wiki_movies_raw)

# %%
# Turn wiki df columns to list
sorted(wiki_movies_df.columns.tolist())

# %%
# Filter data using list comprehension
wiki_movies = [movie for movie in wiki_movies_raw
                if('Director' in movie or 'Directed by' in movie)
                    and 'imdb_link' in movie
                    and 'No. of episodes' not in movie]
len(wiki_movies)

# %%
# Recreate wiki movie df using filtered list
wiki_movies_df = pd.DataFrame(wiki_movies)
# %%
wiki_movies_df[wiki_movies_df['Arabic'].notnull()]

# %%
# Define a function to clean up a movie data by combining alternative titles together in a dict
def clean_movie(movie):
    movie_copy = dict(movie)  #<--Create a non-destructive copy as a dict
    alt_titles = {}  #<--Create an empty dict to hold alternative titles
    for key in ['Also known as', 'Arabic', 'Cantonese', 'Chinese', 'French',
                'Hangul', 'Hebrew', 'Hepburn', 'Japanese', 'Literally',
                'Mandarin', 'McCune-Reischauer', 'Original title', 'Polish',
                'Revised Romanization', 'Romanized', 'Russian',
                'Simplified', 'Traditional', 'Yiddish']:
        if key in movie:  #<--Loop through each alt title key within current movie
            alt_titles[key] = movie[key] #<--Add key-value pair to the alt titles dict
            movie.pop(key) #<--Remove key-value pair from movie object
    if len(alt_titles) > 0: #<--After looping through keys and there are alt titles
        movie['alt_titles'] = alt_titles  #<--Add alt titles dict to movie object
    return movie

# %%
# Run clean_movie function through wiki_movies and set it as Dataframe
clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())

# %%
