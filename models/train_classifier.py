import sys
import pandas as pd 

import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
    database_filepath : Reads sqlite database from database_filepath

    Returns
    -------
    X, y, y.columns: Tuple of Features, target, category_names

    '''
    df = pd.read_sql_table('DisasterResponse1.sql','sqlite:///' + database_filepath)
    X = df['message']
    y = df.drop(['id','message','original','genre'],axis=1)     
    return (X,y,y.columns)

def tokenize(text):
    '''
    

    Parameters
    ----------
    text : for a given text, the function does the following:
        i. Converts all words to lowercase
        ii. Ignores all characters that are not alphabets or numbers.
        iii. Breaks down sentences into words (word_tokenize)
        iv. Removes (Englist) stop_words from the tokens.
        v. Lemmatizes words (reduces each word to its root word)

    Returns
    -------
    lem_tokens : Lemmatized word tokens from the input text.

    '''
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
    '''
    This function does the following: 
        1. Builds a pipeline consisting of 2 steps:
            a. TfidfVectorizer: Which preprocesses the input text string to 
                a set of relevant features which the model can process  
            b. Classifier: Multioutput classifier with randomforest model
        2. GridSearchCV: We then build a GridSearchCV on the pipeline with 
        the required parameter grid (to speed up the training, we only 
        use 1 parameter variation at the moment), and other parameters

    Returns
    -------
    CV : Returns the GridSearchCV model, which can now be fit to training data.

    '''
    pipeline = Pipeline([
        # ('vect', CountVectorizer(tokenizer=tokenize)),
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
         ('rf_multi', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    CV = GridSearchCV(pipeline,
                      param_grid = {'rf_multi__estimator__n_estimators': [50,100]},
                      n_jobs= 1,
                      cv=2)
    return CV

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function prints classification report for each output category 
    for X_test. 

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    Y_test : TYPE
        DESCRIPTION.
    category_names : TYPE
        DESCRIPTION.

    Returns
    -------
    None. 
    '''
    Y_pred = model.predict(X_test)
    
    for colno in range(Y_test.shape[1]):
        print("Target Column = ", Y_test.columns[colno])
        print(classification_report(Y_test.iloc[:,colno],Y_pred[:,colno]))


def save_model(model, model_filepath):
    '''
    The function saves the model as a pickle file, which can be 
    loaded later for predictions. 
    Parameters
    ----------
    model : Fitted model
    model_filepath : location to store the model.

    Returns
    -------
    None.

    '''
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