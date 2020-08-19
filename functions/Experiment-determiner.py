from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math 

import nltk
nltk.download('vader_lexicon')
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

text = "Never, shoo! Shoo! Shoo!"

# Get sentiment
v = SentimentIntensityAnalyzer()
score = v.polarity_scores(text)['compound'] #aggregated score -1 1 
score = (score * 50) + 50
score = round_up(score, 1)
print ("Sentence sentiment: " + str(score))

# Get subjectivity 
subj = TextBlob(text).sentiment #0 objective, 1 subjective
subj = subj[1] * 100
subj = round_up(subj,1)
print ("Sentence subjectivity: " + str(subj))

res = (50,0)
sav = res[0] * res[0] + res[1] * res[1]
sav = pow(sav, 1/2)
scores = [(50,0),(71.1,80)]
ans = []
for ele in scores:
    num = ele[0] * ele[0] + ele[1] * ele[1]
    num = pow(num, 1/2)

    ans.append(abs(sav - num))
print (ans)
print (scores[ans.index(min(ans))])
