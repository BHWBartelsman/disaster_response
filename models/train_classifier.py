import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """
    Args in: database_filepath
    Args out:  X, Y, category_names
    Description: Takes in a filepath and reads in database_file name from process.py
    """
    
    #load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    # split df in X and Y
    X = df['message'].copy()
    Y = df[list(df.columns)[4:]].copy()
    
    # make all y targets binary
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)

    # category names
    category_names = list(Y.columns)

    return X,Y,category_names

def tokenize(text):
    """
    Args in: text file
    Args out: clean lemmatized tokens (words)
    Description: Takes text, splits text in words, deletels urls and lemmatizes these words 
    """

    # finds urls and replaces with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens




def build_model():
    """
    Args in: None
    Args out: model
    Description: build model with pipeline and gridsearch
    """

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # parameters for Gridsearch, if trainingtime is not a problem add uncomment parameters
    parameters = {
#        'vect__ngram_range': ((1, 1), (1, 2)),
#        'vect__max_df': (0.5, 0.75, 1.0),
#        'vect__max_features': (None, 5000, 10000),
#        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
#         'features__transformer_weights': (
#             {'text_pipeline': 1, 'starting_verb': 0.5},
#             {'text_pipeline': 0.5, 'starting_verb': 1},
#             {'text_pipeline': 0.8, 'starting_verb': 1},
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    # Change pipeline to cv if training time is not a problem to optimize model
    return pipeline




def evaluate_model(model, X_test, Y_test, category_names):
    """
    Args in: model, X_test, Y_test, category_names
    Args out: None
    Description: Evaluates every categorie and prints accuracy, recall, precision and F1
    """

    # predict score 
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(data=y_pred, columns = category_names)

    # loop over y and print score 
    for i in category_names:
        print(i)
        print("-------------------------------------------")
        print(classification_report(Y_test[i],y_pred_df[i]))


def save_model(model, model_filepath):
    """
    Args in: model, modelfilepath
    Args out: none
    Description: Saves model on location filepath as pickle
    """

    modelpickle = pickle.dumps(model)
    with open(model_filepath, "wb") as f:
        f.write(modelpickle)

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