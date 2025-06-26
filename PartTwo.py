import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier 
from pathlib import Path
from sklearn.svm import LinearSVC 
from sklearn.metrics import classification_report, f1_score 

import spacy


def spacy_tokenizer(text):
    

    doc = nlp(text)
    content_pos_tags = {"NOUN", "PROPN", "ADJ", "VERB", "ADV", "PRON"}

    # min_token_len = 2
    # max_token_len = 20
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_punct and           
           not token.like_num and
           not token.is_stop and        
           token.pos_ in content_pos_tags 
         ]
    return tokens


def prepare_speech_data(path):
    """
    Read the hansard40000.csv dataset in the texts directory into a dataframe. Sub- 8
    set and rename the dataframe as follows:
    i. rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’, and
    then:
    ii. remove any rows where the value of the ‘party’ column is not one of the
    four most common party names, and remove the ‘Speaker’ value.
    iii. remove any rows where the value in the ‘speech_class’ column is not
    ‘Speech’.
    iv. remove any rows where the text in the ‘speech’ column is less than 1000
    characters long.
    """

    df = pd.read_csv(path,engine='python', on_bad_lines='skip')
    
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

    df = df[df['party'] != 'Speaker']

    party_counts = df['party'].value_counts()
    
    top_4_parties = party_counts.head(4).index.to_list()

    df = df[df['party'].isin(top_4_parties)]

    df = df[df['speech_class'] != 'speech']

    df = df[df['speech'].apply(lambda x: isinstance(x,str) and len(x)>1000 ) ]

    print(df.shape)

    return df

    
def vectorize_split_data(df,with_ngram = False, use_custom_tokenizer=False):
    """
    Vectorise the speeches using TfidfVectorizer from scikit-learn. Use the default
    parameters, except for omitting English stopwords and setting max_features to
    3000. Split the data into a train and test set, using stratified sampling, with a
    random seed of 26.
    """
    random_seed = 26

    tokenizer = spacy_tokenizer if use_custom_tokenizer else None
    if use_custom_tokenizer and with_ngram:
        vectorizer = vectorizer = TfidfVectorizer( ngram_range=(1,3),
        tokenizer=tokenizer,
        min_df=5,      
        max_df=0.999 ,sublinear_tf=True)
    elif not with_ngram:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000,tokenizer=tokenizer)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,3),tokenizer=tokenizer)

    X = vectorizer.fit_transform(df['speech'])
    y = df['party']

    

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

   
 
    return x_train, x_test, y_train, y_test

def train_and_evaluate(x_train, x_test, y_train, y_test):

    """
    Train RandomForest (with n_estimators=300) and SVM with linear kernel clas- 5
    sifiers on the training set, and print the scikit-learn macro-average f1 score and
    classification report for each classifier on the test set. The label that you are
    trying to predict is the ‘party’ value.
    """
    random_seed = 6
    n_estimators = 300

    rf_classifier = RandomForestClassifier(n_estimators=n_estimators,random_state=random_seed)
    svm_classifier = LinearSVC(dual=False ,random_state=random_seed)

    rf_classifier.fit(x_train, y_train)
    svm_classifier.fit(x_train,y_train)

    rf_prediction = rf_classifier.predict(x_test)
    svm_prediction = svm_classifier.predict(x_test)

    print("\n Random forest f1 score:")
    print(f1_score(y_test,rf_prediction,average='macro'))
    print("\n Random forest classification report:")
    print(classification_report(y_test,rf_prediction))

    print("\n SVM f1 score:")
    print(f1_score(y_test,svm_prediction,average='macro'))
    print("\n SVM classification report:")
    print(classification_report(y_test,svm_prediction))




if __name__=="__main__":
    
    nlp = spacy.load('en_core_web_sm')
    data_dir = Path.cwd() / "p2-texts"
    csv_file = data_dir / "hansard40000.csv"

    speech_df =  prepare_speech_data(csv_file)

    # x_train, x_test, y_train, y_test = vectorize_split_data(speech_df)

    # train_and_evaluate(x_train, x_test, y_train, y_test)

    # print('\n\n\n now repeating the proccess with unigrams, bi-grams and tri-grams will be considered as features')

    # x_train, x_test, y_train, y_test = vectorize_split_data(speech_df, True)

    # train_and_evaluate(x_train, x_test, y_train, y_test)

    print('\n\n custome tokenizer ')

    x_train, x_test, y_train, y_test = vectorize_split_data(speech_df,True,True)
    train_and_evaluate(x_train, x_test, y_train, y_test)