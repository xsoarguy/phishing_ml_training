
# %%
import pandas as pd
import numpy as np
import pathlib
import json
import pathlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# fix the pathing for running as a script and as a jupyter notebook
import os, sys;
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import floret

testing_lines = []

with pathlib.Path('testing_output.txt').open() as testing_output_file:
    testing_lines = testing_output_file.readlines()


# %%
import re

input_classes = []
input_emails = []

for line in testing_lines:
    # finds the email data within the testing data lines
    input_emails.append(re.findall(r' (.*$)', line))
    # finds the classifications in the testing data file lines
    input_classes.append(re.match(r'^(.*?) ', line)[1])

# %%
model = floret.load_model('phishing.model')

prediction_results = []
prediction_confidences = []

for email in input_emails:
    # predict with most likely class as output (k=1)
    prediction = model.predict(email, k=1)
    prediction_results.append(prediction[0][0][0])
    prediction_confidences.append(prediction[1][0][0])

# %%
# build confusion matrix
print(model.test_label('testing_output.txt'))
print(model.test('testing_output.txt'))

conf_matrix = confusion_matrix(input_classes, prediction_results, labels=['__label__legit', '__label__phishing'])

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['__label__legit', '__label__phishing'])

import matplotlib.pyplot as plt
disp.plot()
plt.show()


# %%

# find any false negatives that we want to analyze and tune out
false_negatives = []

for idx, prediction_result in enumerate(zip(input_classes, prediction_results)):
    if prediction_result[0] == "__label__phishing" != prediction_result[1]:
        false_negatives.append(input_emails[idx][0])

# %%

import floret
model_file = floret.load_model('phishing.model')

from libs.ml_utils import PhishingNLPExplainer

explainer = PhishingNLPExplainer(model_file)
explainer.explain_prediction(false_negatives[0])
# %%
