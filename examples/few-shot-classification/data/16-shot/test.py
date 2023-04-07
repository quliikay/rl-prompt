import pandas as pd

# download roberta-large's tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
# path = './agnews/16-100/test.tsv'
# path = './sst-2/16-100/test.tsv'
# path = './yelp-2/16-100/test_copy.tsv'
path = './cr/16-100/test.tsv'
df = pd.read_csv(path, sep='\t')
# get the average length of token of df['sentence']
a = 0
for i in range(len(df)):
    a += len(tokenizer(df['sentence'][i])['input_ids'])
a = a / len(df)
print(a)