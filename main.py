import PartOne as p1

#part A

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