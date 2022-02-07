import sys
import pandas as pd
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    ''' 
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on = "id")
    return df

def clean_data(df):
    ''' 
    
    '''
    df['categories'].str.split(';',expand=True)
    categories = df['categories'].str.split(';',expand=True)
    categorieslist = df['categories'].str.split(';',expand=True).loc[0]
    categorieslist = [x[:-2] for x in categorieslist]
    categories.columns = categorieslist
    
    # Convert category numbers into 0 or 1
    for col in categories.columns:
        categories[col] = [x[-1] for x in categories[col]]
        
    # Replace category column in df with new categories
    df.drop(['categories'],axis=1,inplace=True)
    
    df = pd.concat([df,categories],axis=1)

    # Remove Duplicates
    df.drop_duplicates(inplace=True)
    
    return df
    
def save_data(df, database_filename):
    '''
    
    '''
    # filename = os.path.basename(database_filename)
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse1.sql',engine,index=False)

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