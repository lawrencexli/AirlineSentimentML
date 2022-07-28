from DataAnalyzer import DataAnalyzer

my_reader = DataAnalyzer()
my_reader.load_model()
result = my_reader.predict('Took an hour for bags to get out at PHL airport. American Airlines doesn\'t care....'
                           'Delta provides credit if it takes more than 20 minutes. AA just said sorry, but no credit. '
                           'Beyond upsetting!')

print(result)
