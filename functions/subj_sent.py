import torch
import pandas as pd
import argparse
import numpy as np
import math
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import csv



data = open('./30scenes-edited.csv', 'r')
reader = csv.reader(data)

new_csv = "./SLA.csv"

stored_SLA = []
count = 1

v = SentimentIntensityAnalyzer()

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


stored_SLA.append(["sentiment", "subjectivity"])
for row in reader: 
    if count == 1: 
        count +=1 
        continue

    score = v.polarity_scores(row[1])['compound'] #aggregated score -1 1 
    score = (score * 50) + 50


    subj = TextBlob(row[1]).sentiment #0 objective, 1 subjective
    subj = subj[1] * 100

    stored_SLA.append([round_up(score,1), round_up(subj,1)])

with open(new_csv, "w", newline="") as f: 
    writer = csv.writer(f)
    writer.writerows(stored_SLA)
