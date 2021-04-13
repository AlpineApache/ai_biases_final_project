import pandas as pd
from vader_sentiment_library import get_vader_sentiment_score
from text_blob_library import get_text_blob_score
from analysis import t_test_bias_analysis

# get the data
sentences = pd.read_csv("./static-files/equity-evaluation-corpus.csv", delimiter=",")
df = pd.DataFrame(sentences,
				  columns=["Sentence", "Gender", "Race", "Emotion", "ID", "Library", "Negative", "Positive", "Neutral",
						   "Compound"])

# apply data to each library
all_results = []
all_applicable_libraries = [get_vader_sentiment_score, get_text_blob_score]

for library in all_applicable_libraries:
	temp = df.apply(library, axis=1)
	all_results.append(temp)

# export all results to .csv file
results = pd.concat(all_results)
results.to_csv("results.csv")

# run t-tests for race and gender bias on all results
t_test_bias_analysis(results)
