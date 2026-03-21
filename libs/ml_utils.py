import re
import os
import hashlib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import floret
import pandas
import numpy as np
from lime.lime_text import LimeTextExplainer
from logging import Logger as logger

NLTK_DATA_PATH = ""
MODEL_PATH = ""

def nltk_data_init(NLTK_DATA_PATH: str = NLTK_DATA_PATH) -> None:
    '''
    Automatic Dependency Building
    '''
    if NLTK_DATA_PATH:
        NLTK_DATA_PATH = os.path.join(os.getcwd(), NLTK_DATA_PATH)
        os.makedirs(NLTK_DATA_PATH, exist_ok=True)
        nltk.data.path.append(NLTK_DATA_PATH)
        logger.debug(f'NLTK data search paths: {nltk.data.path}')
        try:
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
            logger.debug('NLTK data validated.')
        except LookupError:
            logger.warning('NLTK data not found, attempting to download.')
            try:
                # download the corpus needed to perform tokenization with nltk
                nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)
                nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
                logger.info(
                    f'NLTK data successfully downloaded to {NLTK_DATA_PATH}')
            except Exception as e:
                logger.exception(
                    f'Unable to find and download NLKT data!\n{e}')


EMAIL_REGEX = r'''(?i)(?:[a-z0-9!#$%&'*+/=?^_\x60{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_\x60{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'''
LINK_REGEX = r'/^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$/'
NUM_REGEX = r'\d+(\.\d+)?'


class PhishingNLPExplainer:
    def __init__(self, model: floret.floret._floret = None):
        """
        Initialize with a trained model object.

        Args:
            model (floret.floret._floret): Trained Floret model object
        """
        if model:
            self._model: floret.floret._floret = model
            self._labels = self._model.get_labels()
            self._label_to_index_mapping = {
                label: idx for idx, label in enumerate(self._labels)}
            self._explainer = LimeTextExplainer(
                class_names=self._labels, random_state=72)
        else:
            raise ValueError("model must be provided")

    def _predict_proba(self, texts: list[str] | str):
        """
        Wrapper function that will wrap the floret model prediction return data to the data that the LimeTextExplainer needs to run it's analysis.

        Args:
            texts (list): List of text strings

        Returns:
            numpy.ndarray: Array of shape (n_samples, n_classes) with probabilities
        """
        if isinstance(texts, str):
            texts = [texts]

        probabilities = []

        for text in texts:
            # Get predictions with probabilities
            labels, probs = self._model.predict(text, k=len(self._labels))

            # Create probability array for all classes
            prob_dict = dict(zip(labels, probs))
            prob_array = np.zeros(len(self._labels))

            for label, prob in prob_dict.items():
                if label in self._label_to_index_mapping:
                    prob_array[self._label_to_index_mapping[label]] = prob

            probabilities.append(prob_array)

        return np.array(probabilities)

    def explain_prediction(self, texts: list[str] | str) -> list[tuple]:
        """
        Explains which words are significant to the prediction of the model, and which words are not significant.
        Significant words have weights > 0
        Non-significant words have weights < 0

        Args:
            texts (list | str): List of text strings or single text string

        Returns:
            list: list of tuples containing the word and then it's significance weighting value.
        """
        if not self._explainer:
            raise Exception('Explainer is not available')

        # run the black-box model explainer
        explaination = self._explainer.explain_instance(
            texts, self._predict_proba, num_features=12, num_samples=5000, top_labels=True)
        available_labels = explaination.available_labels()

        if len(available_labels) > 0:
            return explaination.as_list(available_labels[0])
        return []


class PhishingNLPPreProcessor():
    def __init__(self):
        nltk_data_init()
    """Takes in a string and tokenizes it and returns a modified string that has been tokenized in such as way that provides more clarity to the model when training"""

    def _tokenize(self, text: str) -> str:
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

    """Internal function 
    Used to pre-process text Takes in a Pandas Dataframe with subject lines in a column named 'subject' and email bodys in a column called 'body' returns a 
    modified version of the dataframe with a column called text_combined added which contains the pre-processed text ready for training / predictions
    """

    def _preprocess(self, df: pandas.DataFrame) -> pandas.DataFrame:        
        # replace email addresses with placeholder
        df['text_combined'] = df['text_combined'].str.replace(pat=EMAIL_REGEX, repl=' email_placeholder ', regex=True, flags=re.MULTILINE)
        df['text_combined'] = df['text_combined'].str.lower().replace('\n', ' ').replace('\r\n', ' ').replace('\t', ' ').replace('\r', ' ')
        # replace links with placeholder
        df['text_combined'] = df['text_combined'].str.replace(pat=LINK_REGEX, repl='link_placeholder', regex=True, flags=re.MULTILINE)
        # replace number values with placeholder
        df['text_combined'] = df['text_combined'].str.replace(pat=NUM_REGEX, repl='num_placeholder', regex=True, flags=re.MULTILINE)
        # remove non ascii
        df['text_combined'] = df['text_combined'].str.encode('ascii', 'ignore').str.decode('ascii')

        return df

    """Takes in the subject and body text of the email and returns pre-processed text for training"""

    def preprocess_single(self, subject: list[str], body: list[str]) -> str:
        df = pandas.DataFrame([{'subject': subject, 'body': body}])
        preprocessed_text = self._preprocess(df)['text_combined'].iloc[0]
        return preprocessed_text

    """Takes in a Pandas Dataframe with subject lines in a column named 'subject' and email bodys in a column called 'body' returns a 
    modified version of the dataframe with a column called text_combined added which contains the pre-processed text ready for training / predictions
    """

    def preprocess_batch(self, df: pandas.DataFrame) -> pandas.DataFrame:
        preprocessed_text = self._preprocess(df)
        return preprocessed_text


class PhishingNLP():
    def _model_hasher(self, model_path: str = MODEL_PATH,
                      buffer_size: int = (8*1024*1024)) -> str:
        '''
        Creates a SHA256 hash of the model, used to track model performance.
        8MB buffer since we have the memory for it.
        An argument in the future could be made to place this in libs.auth,
        but at the moment there is no need.
        '''
        hasher = hashlib.sha256()
        try:
            with open(model_path, 'rb') as model_file:
                while True:
                    chunk = model_file.read(buffer_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except IOError as e:
            raise IOError(f'Error reading file {MODEL_PATH}: {e}')

    def __init__(self, model_path: str = MODEL_PATH):
        # load model into memory from disk
        self._model = floret.load_model(model_path)
        if self._model:
            self._explainer = PhishingNLPExplainer(self._model)
            self.model_hash = self._model_hasher()

    def predict(self, subject: list[str], body: list[str],
                num_classes_returned: int = 3,
                min_confidence: float = .5,
                on_unicode_error: str = 'strict') -> dict:
        '''
        Is used to perform the prediction for a single email.
        Will return the confidence levels of each classification
        given that num_classes_returned > 1
        '''
        if not self._model:
            raise Exception(
                'Model has not been loaded, cannot make a prediction')

        if not self._explainer:
            raise Exception(
                'Explainer failed to load, cannot make a prediction')

        # must pre-process the text first before predicting
        preprocessor = PhishingNLPPreProcessor()
        pre_processed_text = preprocessor.preprocess_single(
            subject=subject, body=body)

        # actually perform the prediction (this is the juicy stuff)
        prediction = self._model.predict(text=pre_processed_text,
                                         k=num_classes_returned,
                                         threshold=min_confidence,
                                         on_unicode_error=on_unicode_error)

        # explain why the model made it's prediction
        explaination = self._explainer.explain_prediction(
            pre_processed_text)
        details = {'prediction': prediction,
                   'preprocessed': pre_processed_text,
                   'hash': self.model_hash,
                   'explaination': explaination}

        return details
