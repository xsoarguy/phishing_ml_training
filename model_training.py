
# %%
import pandas as pd
import numpy as np
import pathlib
import json
import re
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import nltk
import floret

# fix the pathing for running as a script and as a jupyter notebook
import os, sys  # noqa: E401
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# %%
needed_columns = ['text_combined', 'label']

nigerian_dataset_path = pathlib.Path(current_dir + '/training_data/Nigerian_Fraud.csv') # credit: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
phishing_dataset_path = pathlib.Path(current_dir + '/training_data/phishing_email.csv') # credit: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

resulting_df = pd.DataFrame()

# load the nigerian dataframe
nigerian_dataset_df = pd.read_csv(nigerian_dataset_path.absolute())
# normalize the dataset by making a "text_combined" column in the dataframe to make it easier to merge with the other dataframe
nigerian_dataset_df['text_combined'] = nigerian_dataset_df['subject'].astype(str) + ' ' + nigerian_dataset_df['body'].astype(str)

# load the other phishing dataset
phishing_dataset_df = pd.read_csv(phishing_dataset_path.absolute())

# merge the dataframe on common columns
resulting_df = pd.concat([nigerian_dataset_df, phishing_dataset_df])
    
# trimming down columns on the dataframe
resulting_df = phishing_dataset_df.loc[:, needed_columns]

#%%
EMAIL_REGEX = r'''(?i)(?:[a-z0-9!#$%&'*+/=?^_\x60{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_\x60{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'''
LINK_REGEX = r'/^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$/'
NUM_REGEX = r'\d+(\.\d+)?'

nltk.download('punkt_tab')


"""Internal function 
Used to pre-process text Takes in a Pandas Dataframe with subject lines in a column named 'subject' and email bodys in a column called 'body' returns a 
modified version of the dataframe with a column called text_combined added which contains the pre-processed text ready for training / predictions
"""
# replace email addresses with placeholder
resulting_df['text_combined'] = resulting_df['text_combined'].str.replace(pat=EMAIL_REGEX, repl=' email_placeholder ', regex=True, flags=re.MULTILINE)
resulting_df['text_combined'] = resulting_df['text_combined'].str.lower().replace('\n', ' ').replace('\r\n', ' ').replace('\t', ' ').replace('\r', ' ')
# replace links with placeholder
resulting_df['text_combined'] = resulting_df['text_combined'].str.replace(pat=LINK_REGEX, repl='link_placeholder', regex=True, flags=re.MULTILINE)
# replace number values with placeholder
resulting_df['text_combined'] = resulting_df['text_combined'].str.replace(pat=NUM_REGEX, repl='num_placeholder', regex=True, flags=re.MULTILINE)
# remove non ascii
resulting_df['text_combined'] = resulting_df['text_combined'].str.encode('ascii', 'ignore').str.decode('ascii')

resulting_df.to_pickle('pre_processed.pickle')

# %%
resulting_df = pd.read_pickle('pre_processed.pickle')
# %%
"""Takes in a string and tokenizes it and returns a modified string that has been tokenized in such as way that provides more clarity to the model when training"""
def _tokenize(text: str) -> str:
    stop_words = stopwords.words('english')
    punctuation_chars = string.punctuation
    
    def remove_words(words: list, words_to_remove: list) -> list:
        for word_to_remove in words_to_remove:
            while word_to_remove in words:
                words.remove(word_to_remove)

        return words

    def clean_words(text: str) -> str:
        tokenized_words = word_tokenize(text)
        tokenized_words = remove_words(tokenized_words, stop_words)
        tokenized_words = remove_words(tokenized_words, punctuation_chars)

        return ' '.join(tokenized_words)
    
    return clean_words(text)

# %%
another_df = resulting_df.copy(deep=True)

# perform tokenization
another_df['text_combined'] = another_df['text_combined'].apply(_tokenize)

# %%
# remove duplicates and drop any empty ones
another_df.drop_duplicates(subset=['text_combined'], inplace=True)
another_df.dropna(subset=['text_combined'], inplace=True)

# %%
def create_floret_file(X_data, y_data, path: str) -> str:
    final_file_contents: str = ""
    
    label_mapping = {0: "legit", 1: "phishing"}
    
    for row, X_item in enumerate(X_data):
        final_file_contents += f'__label__{label_mapping.get(y_data.iloc[row])} {X_item}\n'

    pathlib.Path(path).write_text(final_file_contents)

from sklearn.model_selection import train_test_split  # noqa: E402

# we want to split up our training data so that we use 80% for training, and 20% for testing after training to see what the accuracy of inferences to the model are
# we also want to shuffle the data around before we train to remove any biases
X_train, X_test, y_train, y_test = train_test_split(another_df['text_combined'], another_df['label'], test_size=0.20, random_state=42, shuffle=True, stratify=None)


# create the training and testing data files
create_floret_file(X_train, y_train, 'training_output.txt')
create_floret_file(X_test, y_test, 'testing_output.txt')


# %%
import floret  # noqa: E402, F811
import pathlib  # noqa: E402

model = floret.train_supervised(input='training_output.txt', epoch=1, lr=.002, wordNgrams=5, verbose=True, seed=42)
#model = floret.train_supervised(input='training_output.txt', autotuneValidationFile='testing_output.txt', autotuneDuration=7200, autotuneModelSize="900M", seed=42)
test_data = str(model.test_label('testing_output.txt'))
pathlib.Path('testing_results.txt').write_text(test_data)

# %%
model.save_model('phishing.model')
# %%
import json  # noqa: E402, F811, F401
import floret  # noqa: E402

def get_hyperparameter_dict(args):
    result_dict = dict()
    
    attributes = dir(args)
    for attribute in attributes:
        if not attribute.startswith('_'):
            result_dict[attribute] = getattr(args, attribute)
    return result_dict

model = floret.load_model('phishing.model')

print(get_hyperparameter_dict(model.f.getArgs()))
