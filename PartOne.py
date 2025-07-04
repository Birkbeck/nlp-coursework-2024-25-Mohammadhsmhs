#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import math
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
import spacy
from pathlib import Path
import pandas as pd
import glob
import os
from collections import Counter


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

nltk.download('punkt_tab')
# nltk.download('cmudict')


def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    if num_sentences == 0:
        return 0.0 

    
    total_words = 0
    total_syllables = 0

    for sentence in sentences:
        
        words = [word.lower() for word in word_tokenize(sentence) if word.isalpha()]
        total_words += len(words)

        for word in words:
            total_syllables += count_syl(word, d) 

    if total_words == 0:
        return 0.0 

    score = (0.39 * (total_words / num_sentences)) + \
            (11.8 * (total_syllables / total_words)) - 15.59

    return score



def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word_lower = word.lower()
    
    if word_lower in d:

        num_vowels = 0
        for phonem in d[word_lower][0]:
            if phonem[-1].isdigit():
                num_vowels+=1
        return max(1,num_vowels)
    else:
        count =0
        vowels = 'aeiouy'
        word_lower = word.lower()
        if word_lower[0] in vowels:
            count+=1
            for index in range(1, len(word_lower)):
                if word_lower[index] in vowels and word_lower[index - 1] not in vowels:
                    count += 1
        
        if word_lower.endswith("e"):
            count -= 1
        if word_lower.endswith("le") and len(word_lower) > 2 and word_lower[-3] not in vowels:
            count += 1
        return max(1, count) 


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""

    novel_data = []

    directory_path = Path(path)

    if not directory_path.exists():
        print(f"Error: Directory not found at '{directory_path}'")
        return pd.DataFrame(columns=['text', 'title', 'author', 'year'])
    if not directory_path.is_dir():
        print(f"Error: Path '{directory_path}' is not a directory.")
        return pd.DataFrame(columns=['text', 'title', 'author', 'year'])
    
    txt_files = glob.glob(os.path.join(directory_path,'*.txt'))
    
    for novel in txt_files:
        with open(novel,'r') as file:
            content = file.read()

            file_name = os.path.basename(novel)
            
            file_name_noext = os.path.splitext( file_name)[0]
            parts = file_name_noext.split('-')
            
            year = int(parts[-1])
            author = parts[-2]
            title = parts[:-2]
            print(title)
            title = '_'.join(title)
            title =title.replace('_',' ')
        
        novel_data.append({
            'text': content,
            'title': title,
            'author': author,
            'year': year
        })

    
    df = pd.DataFrame(novel_data)
    
    df_sorted = df.sort_values(by='year', ascending=True).reset_index(drop=True)

    return df_sorted

    


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

    pickle_filepath = store_path / out_name
    print(out_name)
    print(pickle_filepath)

    store_path.mkdir(parents=True, exist_ok=True)

    df['parsed'] = list(nlp.pipe(df['text']))

    df.to_pickle(pickle_filepath)

    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""


    tokens = word_tokenize(text)

    processed_tokens = []

    for token in tokens:
        token_lower = token.lower()

        if token_lower.isalpha():
            processed_tokens.append(token_lower)
    # print(f"Processed tokens (lowercase, no punctuation): {processed_tokens}")
    if len(processed_tokens)>0:
        num_tokens = len(processed_tokens)
        num_types = len(set(processed_tokens))
        ttr= num_types/num_tokens
        return ttr
    else:
        return 0

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    
    total_tokens_in_doc = len([token for token in doc if token.is_alpha])

    verb_counts = Counter(token.lemma_.lower() for token in doc if token.pos_ == "VERB" and token.is_alpha)

    count_y = verb_counts.get(target_verb,0)

    if count_y ==0:
        return []
    
    subject_verb_co_occurrences = Counter()
    all_alphabetic_tokens = Counter(token.lemma_.lower() for token in doc if token.is_alpha)

    for token in doc:
        if token.lemma_.lower() == target_verb and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ['nsubj','nsubjpass','agent'] and child.is_alpha:
                    subject_lemma = child.lemma_.lower()
                    subject_verb_co_occurrences[subject_lemma] +=1
    pmi_scores = {}

    for subject_lemma, count_xy in subject_verb_co_occurrences.items():
        count_x = all_alphabetic_tokens.get(subject_lemma,0)

        if count_x > 0 and count_y > 0:
            pmi = math.log2((count_xy * total_tokens_in_doc) / (count_x * count_y))
            pmi_scores[subject_lemma] = pmi

    sorted_pmi_scores = sorted(pmi_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_pmi_scores[:10]



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    
    subjects = []

    for token in doc:
        if token.lemma_.lower() == verb and token.pos_ =="VERB":
            for child in token.children:
                if child.dep_ in ['nsubj','nsubjpass','agent'] and child.is_alpha:
                    subjects.append(child.text.lower())

    return Counter(subjects).most_common(10)
    



def common_syntatic_objects(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    objects = []

    object_deps = {"dobj","iobj","pobj"}
    for token in doc:
        if token.dep_ in object_deps and token.is_alpha:
            objects.append(token.text.lower())

    return Counter(objects).most_common(10)



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    # parse(df, out_name='name.pickle')
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    print("\n printing common syntatic objects")
    for i,row in df.iterrows():
        print(row['title'])
        print(row)
        print(common_syntatic_objects(row['parsed']))
        print("\n")

    print("\n printing subjects by verb count")
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")
    print("\n printing subjects by verb pmi")
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    

