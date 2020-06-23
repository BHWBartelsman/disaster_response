import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ 
    Args in: filepath for messages,filtepath for categories.
    Args out: df
    Description: Functions takes to filepaths, reads in csv's and merges on index
    """
    
    # read messages and categorie from file path from file paths
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merges data sets
    df = messages.merge(categories, on = 'id')

    return df 


def clean_data(df):
    """
    Args in: df
    Agrs out: df
    Description: Takes a df, cleans categories column and splits it to separate columns, returns this df
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ";", expand = True)

    # rename the columns of `categories`
    row = categories[:1].copy()
    category_colnames = pd.np.array(row.applymap(lambda x: x[:-2])).tolist()[0]
    categories.columns = category_colnames

    # convert catergoy values to numbers 0 or 1

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1:]).astype('int64')

    # drop the original categories column from `df`
    df.drop('categories', inplace = True, axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index = True, right_index = True)

    # drop duplicate values
    df.drop_duplicates(subset = list(df.columns)[1:], inplace = True)

    return df



def save_data(df, database_filename):
    """
    Args in: takes an df and database_filename
    Args out: None
    Description: Saves database into sql lite db
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql(database_filename[:-3], engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()