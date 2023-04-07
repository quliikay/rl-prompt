import pandas as pd

path = 'test_copy.tsv'
df = pd.read_csv(path, sep='\t')
# get 2000 rows of df with random state 42
df = df.sample(n=2000, random_state=42)
df.to_csv('./test.tsv', sep='\t', index=False)