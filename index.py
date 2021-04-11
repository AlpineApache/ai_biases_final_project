import pandas as pd
from vader_sentiment_library import get_vader_sentiment_score
from text_blob_library import get_text_blob_score

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

results = pd.concat(all_results)
results.to_csv("results.csv")

# aggregate mean compound value for all categories along these parameters: library x gender x race x emotion
results_list = []
libraries = ["vader_sentiment", "text_blob"]
emotions = ["anger", "joy", "fear", "sadness"]
genders = ["female", "male"]
races = ["European", "African-American"]

for library in libraries:
  for emotion in emotions:
    for gender in genders:
      for race in races:
        temp_df = results.loc[(results["Library"]==library) & (results["Gender"]==gender) & (results["Emotion"]==emotion) & (results["Race"]==race)]
        temp_data = {
            "Library": library,
            "Gender": gender,
            "Race": race,
            "Emotion": emotion,
            "Mean": temp_df["Compound"].mean()
        }
        results_list.append(temp_data)
      temp_df = results.loc[(results["Library"]==library) & (results["Gender"]==gender) & (results["Emotion"]==emotion) & (results["Race"]!="European") & (results["Race"]!="African-American")]
      temp_data = {
          "Library": library,
          "Gender": gender,
          "Race": "na",
          "Emotion": emotion,
          "Mean": temp_df["Compound"].mean()
      }
      results_list.append(temp_data)

aggregated_results = pd.DataFrame(results_list)
aggregated_results.to_csv("aggregated_results.csv")
