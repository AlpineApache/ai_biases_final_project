from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def get_vader_sentiment_score(data):
    data.Library = 'vader_sentiment'
    score = analyser.polarity_scores(data.Sentence)
    data.Positive = score['pos']
    data.Negative = score['neg']
    data.Neutral = score['neu']
    data.Compound = score['compound']
    return data

