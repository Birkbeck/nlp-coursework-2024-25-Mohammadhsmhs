import pandas as pd 
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 


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

    df = pd.read_csv(path)
    
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

    df = df[df['party'] != 'Speaker']

    party_counts = df['party'].value_counts()
    
    top_4_parties = party_counts.head(4).index.to_list()

    df = df[df['party'].isin(top_4_parties)]

    df = df[df['speech_class'] != 'speech']

    df = df[df['speech'].apply(lambda x: isinstance(x,str) and len(x)>1000 ) ]

    print(df.shape)

    return df

    
def vectorize_split_data(df):
    """
    Vectorise the speeches using TfidfVectorizer from scikit-learn. Use the default
    parameters, except for omitting English stopwords and setting max_features to
    3000. Split the data into a train and test set, using stratified sampling, with a
    random seed of 26.
    """
    random_seed = 26
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']

    

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

   
 
    return x_train, y_train, x_test, y_test



if __name__=="__main__":
    
    
    data_dir = Path.cwd() / "p2-texts"
    csv_file = data_dir / "hansard40000.csv"

    speech_df =  prepare_speech_data(csv_file)

    x_train, y_train, x_test, y_test = vectorize_split_data(speech_df)

    