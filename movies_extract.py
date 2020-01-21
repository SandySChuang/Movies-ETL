# %%
# Import dependencies
import json 
import pandas as pd 
import numpy as np 
import re 


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

    # merge similar column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by','Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')
    
    return movie

# %%
# Run clean_movie function through wiki_movies and set it as Dataframe
clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())

# %%
# Extract imdb_id using regex from the dataframe and create a new column for joining
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
# Drop duplicate imdb_id
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()

# %%
# Check for nulls in each column using list comprehension
[[column, wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]

# %%
# Narrow down to columns to keep, with less than 90% null
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
wiki_movies_df.columns.tolist()

# %%
# Inspect column data type for potential conversion
wiki_movies_df.dtypes
# %%
# Clean Box Office data
box_office = wiki_movies_df['Box office'].dropna()


# %%
# Create a function using regex to first make sure box office is entered 
# as string before conversion

def is_not_a_string(x):
    return type(x) != str  #<--will be replaced by lambda function

# %%
# See which box office data point is not a string
box_office[box_office.map(is_not_a_string)]

# %% 
# using lamda function instead of is_not_a_string function
box_office[box_office.map(lambda x: type(x) != str)]

# Output show many data points are stored as list.

# %%
# Join the list using a space as separator
box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

# %%
# define possible form of box office data using regex
form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)' #<-after allowing comma/period as separator need to lookahead and exclude [mb]illion

# %%
# Replace values given as a range with the upper range
box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
# %%
# See how many matches form one using str.contains() to ignore cases
box_office.str.contains(form_one, flags=re.IGNORECASE).sum()

# %%
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()

# %%
# check which data points do not match either form
matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

# check for those not matching both using element-wise operators
box_office[~matches_form_one & ~matches_form_two]

# %%
# define function to convert box office value
def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan

# Extract box office string and apply function, and drop the original column
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
wiki_movies_df.drop('Box office', axis=1, inplace=True)

# %%
# Create a budget variable
budget = wiki_movies_df['Budget'].dropna()

# %%
# Convert list into strings
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

# %%
# Fixing budgets given in ranges
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

# %%
# check which data points do not match either form
matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)

# check for those not matching both using element-wise operators
budget[~matches_form_one & ~matches_form_two]

# %%
# Remove citation [#] reference from budget data
budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]


# %%
# Extract budget string and apply function, and drop the original column
wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
wiki_movies_df.drop('Budget', axis=1, inplace=True)

# %%
# Create release date variable
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

# %%
# Define different date forms
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'

# Extract date
wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


# %%
# Create running time variable
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

# %%
# Check running time format
running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()

# %%
# Check running time data in other format
running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]

# %%
# Extract running time into string
running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

# %%
# Convert string into numeric
running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

# %%
# Convert to minutes
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# %%
# Drop old running time column
wiki_movies_df.drop('Running time', axis=1, inplace=True)

# %%
# Check Kaggle data
kaggle_metadata.dtypes

# %%
# Check adult values consistent with data type
kaggle_metadata['adult'].value_counts() #<-initial output contains bad data, not just boolean

# %%
# Remove bad kaggle data in adult column as well as adult movies
kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')


# %%
# Check video values consistent with data type
kaggle_metadata['video'].value_counts()


# %%
# Convert to boolean by comparing string to 'True'
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# %%
# Convert to numeric
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')

# %%
# Convert release date
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# %%
# Reasonability checks on ratings data
ratings.info(null_counts=True)

# %%
# Converting epoch timestamp
pd.to_datetime(ratings['timestamp'], unit='s')

# %%
# Reassign to 'timestamp'
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], origin="unix", unit='s')

# %%
# Check rating distribution
ratings['rating'].describe()

# %%
ratings['rating'].plot(kind='hist')

# %%
# Check columns for merging
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

# %%
# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle             Drop Wiki
# running_time             runtime                  Kaggle; fill zeroes with Wiki
# budget_wiki              budget_kaggle            Kaggle; fill zeroes with Wiki
# box_office               revenue                  Kaggle; fill zeroes with Wiki
# release_date_wiki        release_date             Drop Wiki
# Language                 original_language        Drop Wiki
# Production company(s)    production_companies     Drop Wiki

# %%
# Check title
movies_df[['title_wiki','title_kaggle']]

# %%
# Check titles where they're different
movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]

# %%
# Check missing titles in Kaggle
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]

# %%
# Check consistency across two runtime data points, with null filled in as zeros
movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')

# %%
# Check consistency across two budget data
movies_df.fillna(0).plot(x='budget_wiki', y='budget_kaggle', kind='scatter')

# %%
# Check consistency across two box office data
movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')

# %%
# Check consistency across two box office data < 1 billion
movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')

# %%
# Check consistency across two release date data.  Use line plots marker only for date data.
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')

# %%
# Investigate 2006 outlier
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


# %%
# Find index of the problematic row - 'The Holiday' got merged with 'From Here to Eternity'
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index

# %%
# Drop identified row
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# %%
# Check for null release data - not plotted
movies_df[movies_df['release_date_wiki'].isnull()]

# %%
# Check for null release data - not plotted
movies_df[movies_df['release_date_kaggle'].isnull()]

# %%
# Check wiki language.  Some data points are list so convert to tuple before value_counts()
movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# %%
# Check Kaggle language
movies_df['original_language'].value_counts(dropna=False)

# %%
# Check Production Company data
movies_df[['Production company(s)','production_companies']]

# %%
# Drop unwanted Wiki columns
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# %%
# Fill missing data function
def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)

# %%
# Fill missing values
fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df

# %%
# Check merged columns that only has one value
for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)

# %%
# check video column value
movies_df['video'].value_counts(dropna=False)

# %%
# Skill drill using list comprehension
[col for col in movies_df.columns if (len(movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)) == 1)]

# %%
# Reorder columns
movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]
# %%
# Rename columns
movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)
                 