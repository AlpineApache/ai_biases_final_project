# Decoding Biases in Artificial Intelligence: Testing Python Sentiment Analysis Libraries for Racial and Gender Biases
by Nicolaus CHRISTENSON, Matteo NEBBIAI & Orestas STRAUKA

## The Paper
The final version of the paper can be found [here](https://drive.google.com/file/d/1yQGquzA9vVzd0UpjHrCDgVwclhf9K1oi/view?usp=sharing).

## Requirements
In order to run the code yourself, make sure you follow these steps:

 - install Python 3.6 on your machine.
 - run `pip install requirements.txt` in your console.
 - run `python index.py` in your console.

## The Files
- **index.py** = the main script that has to be called and invokes all of the relevant functions.
- **equity_evaluation_corpus.csv** = the file containing the test sentences from the [Equity Evaluation Corpus](https://www.svkir.com/resources/Equity-Evaluation-Corpus.zip) with adjusted column names and some redactions/additions.
- **text_blob_library.py** = the script that returns the sentiment analysis score of the [TextBlob library](https://textblob.readthedocs.io/en/dev/) for each sentence.
- **vader_sentiment_library.py** = the script that returns the sentiment analysis score of the [VADERSentiment library](https://github.com/cjhutto/vaderSentiment) for each sentence.
- **results.csv** = the file containing the data from equity_evaluation_corpus.csv and the scores of each library for each sentence.
- **analysis.py** = the file that runs statistical analysis on results.csv along racial and gender lines.
- **gender_bias_results.csv** = the file containing the results of analysis.py on gender bias in the libraries.
- **racial_bias_results.csv** = the file containing the results of analysis.py on racial bias in the libraries.
- **mean_differences.png** = the file containing the plot of the differences between the mean predicted scores of clusters of sentence pairs.
- **requirements.txt** = the file containing all of the dependencies.