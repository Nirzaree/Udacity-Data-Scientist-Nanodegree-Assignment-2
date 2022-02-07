import sys
import pandas as pd 

import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    Parameters
    ----------
    database_filepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df = pd.read_sql_table('DisasterResponse1.sql','sqlite:///' + database_filepath)
    X = df['message']
    y = df.drop(['id','message','original','genre'],axis=1)     
    return (X,y,y.columns)

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9]',' ',text)
    text = word_tokenize(text)
    stop_words = stopwords.words('english')
    text = [x for x in text if x not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lem_tokens = []
    for token in text:
        lem_tokens.append(lemmatizer.lemmatize(token))
    return lem_tokens   


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
         ('rf_multi', MultiOutputClassifier(RandomForestClassifier()))
        ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    class_report = classification_report(y_true = Y_test,
                                         y_pred = Y_pred,
                                         target_names=category_names)
    return class_report


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    # pickle.dump(model,open(model_filepath),'wb')
    return 

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()