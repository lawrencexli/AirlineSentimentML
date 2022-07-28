import re
import string
import random
import pickle

import pandas as pd
from nltk import WordNetLemmatizer, NaiveBayesClassifier, classify
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer


class DataAnalyzer:

    def __init__(self, stop_words=stopwords.words('english')):
        self.positive_tokens = [] # All word tokens for positive data
        self.negative_tokens = [] # All word tokens for negative data
        self.positive_cleaned_tokens = [] # All cleaned word tokens (no hyperlinks and special characters) for positive data that 
        self.negative_cleaned_tokens = [] # All cleaned word tokens (no hyperlinks and special characters) for negative data that 
        self.dataframe = None # The dataframe for reading the csv datafile
        self.positive_dataset = None # Positive dataset in a format that can be fed into a nltk modl
        self.negative_dataset = None # Negative dataset in a format that can be fed into a nltk modl
        self.training_dataset = None # Dataset for taining
        self.testing_dataset = None # Dataset for testing
        self.stop_words = stop_words # Stop words used for cleaning the word tokens (default in english)
        self.tweet_tokenizer = TweetTokenizer() # A tokenizer specializes in tokenizing tweet sentences
        self.classifier = None # The main classifier model

    def read_from_file(self, filepath: str):
        # Must be csv file and valid file path
        assert len(filepath) >= 5 and filepath[-3:] == 'csv', "Please enter valid filepath (csv format only)"

        self.dataframe = pd.read_csv(filepath)

    """
    Tokenize sentences and split it into word tokens
    """
    def tokenize_msg(self):
        assert self.dataframe is not None, "Data has not been read yet"

        # Reset tokens
        self.positive_tokens = []
        self.negative_tokens = []

        for i in range(len(self.dataframe)):
            if self.dataframe.airline_sentiment[i] == 'positive':
                self.positive_tokens.append(self.tweet_tokenizer.tokenize(self.dataframe.text[i]))
            elif self.dataframe.airline_sentiment[i] == 'negative':
                self.negative_tokens.append(self.tweet_tokenizer.tokenize(self.dataframe.text[i]))
            else:
                raise ValueError('Dataset contains invalid categories: ' + self.dataframe.airline_sentiment[i])

    """
    Remove any punctuations, special characters, and hyperlinks from the list of tokens
    """
    def token_remove_noise(self):
        # Reset tokens
        self.positive_cleaned_tokens = []
        self.negative_cleaned_tokens = []

        for tokens in self.positive_tokens:
            self.positive_cleaned_tokens.append(self.remove_noise(tokens))

        for tokens in self.negative_tokens:
            self.negative_cleaned_tokens.append(self.remove_noise(tokens))

    """
    Organize the format of dataset to be fed into the nltk model
    """
    def organize_dataset_for_model(self):
        positive_tokens_for_model = self.get_tweets_for_model(self.positive_cleaned_tokens)
        negative_tokens_for_model = self.get_tweets_for_model(self.negative_cleaned_tokens)

        self.positive_dataset = [(tweet_dict, "Positive")
                                 for tweet_dict in positive_tokens_for_model]

        self.negative_dataset = [(tweet_dict, "Negative")
                                 for tweet_dict in negative_tokens_for_model]

    """
    Split train and test data by given ratio
    """
    def train_test_split(self, train_test_ratio=8):
        assert self.positive_dataset is not None and self.negative_dataset is not None, "Dataset is not organized"

        random.shuffle(self.positive_dataset)
        random.shuffle(self.negative_dataset)

        positive_train_data = self.positive_dataset[len(self.positive_dataset) // train_test_ratio:]
        positive_test_data = self.positive_dataset[:len(self.positive_dataset) // train_test_ratio]
        negative_train_data = self.negative_dataset[len(self.negative_dataset) // train_test_ratio:]
        negative_test_data = self.negative_dataset[:len(self.negative_dataset) // train_test_ratio]

        self.training_dataset = positive_train_data + negative_train_data
        self.testing_dataset = positive_test_data + negative_test_data

    """
    Remove hyperlinks, punctuations and special characters from the word tokens
    """
    def remove_noise(self, tweet_tokens):
        cleaned_tokens = []

        for token, tag in pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*(),]|''(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                           token)
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in self.stop_words:
                cleaned_tokens.append(token.lower())

        return cleaned_tokens

    """
    Train the model by given number of epochs. Save the model with highest testing accuracy
    """
    def train(self, num_epochs=20):
        assert self.positive_dataset is not None and self.negative_dataset is not None, "Dataset is not loaded"

        print("Amount of training data: ", len(self.training_dataset))
        print("Amount of testing data:", len(self.testing_dataset))

        max_accuracy = 0
        for _ in range(num_epochs):
            classifier_candidate = NaiveBayesClassifier.train(self.training_dataset)
            curr_accuracy = classify.accuracy(classifier_candidate, self.testing_dataset)

            if curr_accuracy > max_accuracy:
                self.classifier = classifier_candidate
                max_accuracy = curr_accuracy

        assert self.classifier is not None, "Training is not properly done"
        print("Accuracy is:", classify.accuracy(self.classifier, self.testing_dataset))
        print(self.classifier.show_most_informative_features(10))

    """
    Predict the tweet message.
    Input: input tweet message
    Output: Sentiment result, either "Positive" or "Negative"
    """
    def predict(self, input_tweet: str) -> str:
        assert self.classifier is not None, "Model is not trained or loaded"

        input_tokens = self.remove_noise(self.tweet_tokenizer.tokenize(input_tweet))
        sentiment_result = self.classifier.classify(dict([token, True] for token in input_tokens))

        return sentiment_result

    """
    Save model to file
    """
    def save_model(self, save_path='../model/my_classifier.pickle'):
        f = open(save_path, 'wb')
        pickle.dump(self.classifier, f)
        f.close()

    """
    Load model from file
    """
    def load_model(self, load_path='../model/my_classifier.pickle'):
        f = open(load_path, 'rb')
        self.classifier = pickle.load(f)
        f.close()

    """
    Convert tweet tokens to dataset format that are ready to be fed into the nltk model
    """
    @staticmethod
    def get_tweets_for_model(cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)
