from pattern3.text.en import sentiment

def get_pattern_score(data):
	data.Library = 'pattern'
	data.Compound = sentiment(data.Sentence)[0]
	return data
