import pandas as pd
from vader_sentiment import get_vader_sentiment_score
from text_blob import get_text_blob_score
from pattern import get_pattern_score

# get the data
sentences = pd.read_csv('./static-files/equity-evaluation-corpus.csv', delimiter=',')
df = pd.DataFrame(sentences,
				  columns=['Sentence', 'Gender', 'Race', 'Emotion', 'ID', 'Library', 'Negative', 'Positive', 'Neutral',
						   'Compound'])

# results = pd.DataFrame(columns=['Sentence', 'Gender', 'Race', 'Emotion', 'ID', 'Library', 'Negative', 'Positive', 'Neutral', 'Compound'])

# apply data to each library
all_results = []
all_libraries = [get_vader_sentiment_score, get_text_blob_score, get_pattern_score]

for library in all_libraries:
	temp = df.apply(library, axis=1)
	all_results.append(temp)

results = pd.concat(all_results)
results.to_csv("results.csv")
