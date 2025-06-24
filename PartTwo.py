import pandas as pd 
from pathlib import Path


def prepare_speach_data(path):
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

    



if __name__=="__main__":
    
    
    data_dir = Path.cwd() / "p2-texts"
    csv_file = data_dir / "hansard40000.csv"

    prepare_speach_data(csv_file)