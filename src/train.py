from DataAnalyzer import DataAnalyzer

my_reader = DataAnalyzer()
my_reader.read_from_file('../data/airline_sentiment_analysis.csv')
my_reader.tokenize_msg()
my_reader.token_remove_noise()
my_reader.organize_dataset_for_model()
my_reader.train_test_split()
my_reader.train()
my_reader.save_model()

