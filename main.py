import PartOne as p1

from nltk.corpus import cmudict

#part A
cmu_dict = cmudict.dict()
read_novels_df = p1.read_novels(path='p1-texts/novels')

print(read_novels_df)

# part B

ttr_ratios = {}

for index, row in read_novels_df.iterrows():
    title = row['title']
    text = str(row['text'])
    ttr = p1.nltk_ttr(text=text)
    ttr_ratios[title]= ttr

for title,ttr in ttr_ratios.items():
    print(f'{title}: {ttr}')


# part C

fk_scores = {}  
        
for index, row in read_novels_df.iterrows():
    title = row['title']
    text = str(row['text'])
    
    fk = p1.fk_level(text, cmu_dict)
    fk_scores[title] = fk

print("\nFlesch-Kincaid Scores:")
for title, score in fk_scores.items():
    print(f"- {title}: {score:.2f}")