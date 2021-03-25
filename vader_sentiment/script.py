from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

countries = pd.read_csv('../static_files/countries.csv')
df = pd.DataFrame(countries, columns= ['Country'])
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(country):
    score = analyser.polarity_scores(f'There is war in ${country}.')
    #print("{:-<40} {}".format(sentence, str(score)))
    print(f'{country} --- {str(score)}')

df.Country.apply(sentiment_analyzer_scores)

