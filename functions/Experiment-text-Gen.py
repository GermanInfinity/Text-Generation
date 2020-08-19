from keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Prob import Prob
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from pickle import dump, load
from textblob import TextBlob
import math

import nltk
nltk.download('vader_lexicon')

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

### TOKENIZE
def read_file(filepath):
	with open(filepath) as f:
		str_text = f.read()
	return str_text

text = read_file('./data_full.txt')
tokens = text.split(" ")
print (len(tokens))
tokens.pop(0)

train_len = 3+1
text_sequences = []
for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

sequences = {}
count = 1
for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences) 

#Collecting some information   
unique_words = tokenizer.index_word
unique_wordsApp = tokenizer.word_counts
vocabulary_size = len(tokenizer.word_counts)
dump(tokenizer,open('tokenizer_Model4','wb'))  
tokenizer = load(open('./tokenizer_Model4','rb'))

text = "Shit! Let’s umm, umm…"

## EXPERIMENT
#Sentiment
sentiment = "68.9 50.0 50.0 50.0 61.4 50.0 64.8 50.0 64.8 50.0 35.2 26.2 50.0 35.2 71.1 50.0 50.0 50.0 17.6 17.6 26.2 37.5 50.0 64.8 74.7 50.0 53.9 35.2 22.9 50.0 50.0 50.0 95.3 46.2 20.8 22.2 50.0 50.0 63.7 50.0 35.2 50.0 30.0 24.5 50.0 50.0 50.0 20.8 50.0 13.4 52.6 86.8 27.1 55.2 36.4 50.0 46.2 50.0 75.6 63.7 85.1 73.9 73.9 50.0 92.8 87.2 91.1 77.2 31.6 64.8 16.0 50.0 50.0 44.9 50.0 48.8 50.0 28.0 50.0 50.0 22.2 46.2 56.4 50.0 64.8 50.0 50.0 8.0 55.2 50.0 50.0 50.0 50.0 50.0 30.0 72.1 50.0 19.1 50.0 50.0 22.9 72.1 50.0 31.9 50.0 43.6 50.0 11.5 50.0 50.0 19.4 50.0 19.4 26.6 50.0 50.0 30.0 50.0 50.0 26.2 68.1 51.3 71.1 50.0 50.0 9.9 16.0 37.5 50.0 70.1 22.2 50.0 20.1 10.2 20.1 50.0 50.0 50.0 6.9 50.0 50.0 22.2 72.7 50.0 50.0 67.0 9.9 35.2 20.8 68.1 50.0 44.9 50.0 22.9 30.0 50.0 50.0 63.7 50.0 61.4 50.0 50.0 50.0 50.0 50.0 16.0 50.0 64.8 50.0 33.0 50.0 50.0 50.0 29.2 50.0 50.0 50.0 50.0 47.5 47.5 74.7 74.7 53.9 32.0 50.0 35.2 50.0 50.0 79.3 50.0 50.0 50.0 74.7 88.8 50.0 14.2 91.7 91.4 50.0 42.4 42.4 64.8 72.1 6.8 63.7 50.0 38.7 5.5 63.7 70.0 73.9 35.2 50.0 15.5 61.4 72.1 81.9 50.0 81.9 14.7 50.0 23.7 83.0 70.1 70.1 21.5 65.5 94.5 50.0 86.4 85.7 53.2 72.1 73.9 50.0 76.3 66.0 50.0 50.0 87.9 64.8 50.0 50.0 68.1 32.0 57.7 80.7 39.9 70.1 66.0 70.9 81.3 50.0 41.0 64.8 43.0 50.0 50.0 50.0 92.1 60.2 92.3 81.9 50.0 90.7 50.0 47.5 50.0 58.9 39.9 67.0 50.0 50.0 50.0 50.0 50.0 85.5 50.0 73.0 32.1 50.0 50.0 53.9 50.0 50.0 50.0 90.9 69.1 50.0 68.1 74.4 50.0 27.2 94.5 42.5 50.0 50.0 38.0 88.5 50.0 50.0 50.0 75.6 23.7 50.0 50.0 50.0 25.4 50.0 68.1 50.0 50.0 75.8 50.0 81.9 76.4 50.0 81.3 74.7 22.2 50.0 50.0 50.0 50.0 50.0 50.0 84.9 61.4 53.9 50.0 50.0 46.2 50.0 80.6 50.0 50.0 50.0 50.0 50.0 50.0 95.9 28.0 93.5 44.9 38.7 15.3 33.0 55.2 63.0 50.0 68.1 67.0 50.0 4.1 24.6 71.0 19.0 77.8 75.6 68.1 35.2 50.0 63.7 50.0 81.3 87.9 76.4 75.2 11.5 92.6 43.6 47.5 50.0 19.8 26.2 21.5 5.6 90.1 50.0 36.4 22.9 64.8 31.0 20.5 93.2 50.0 50.0 81.3 50.0 97.4 63.6 50.0 50.0 65.5 22.2 47.5 50.0 50.0 14.8 32.0 80.0 71.1 75.6 50.0 50.0 50.0 42.4 13.3 80.7 93.6 66.0 50.0 50.0 30.0 35.2 31.4 50.0 80.7 67.0 50.0 23.7 31.0 12.7 50.0 50.0 53.9 86.4 50.0 64.8 50.0 50.0 23.7 65.5 50.0 61.4 72.1 50.0 50.0 29.0 50.0 50.0 26.2 81.9 50.0 50.0 50.0 50.0 50.0 50.0 50.0 16.0 50.0 50.0 50.0 50.0 64.8 26.2 22.2 50.0 16.0 39.2 35.2 50.0 17.6 63.7 27.1 70.1 50.0 50.0 50.0 50.0 16.0 50.0 64.8 50.0 50.0 50.0 50.0 22.2 35.2 50.0 22.2 50.0 50.0 50.0 50.0 77.6 63.7 50.0 22.9 22.9 22.9 50.0 50.0 50.0 23.7 50.0 50.0 50.0 19.7 50.0 50.0 50.0 50.0 50.0 68.1 32.1 50.0 50.0 53.9 17.1 50.0 50.0 53.9 28.0 12.2 50.0 50.0 53.9 64.8 50.0 50.0 50.0 20.3 50.0 50.0 50.0 61.4 50.0 35.2 58.9 5.7 50.0 22.9 29.0 63.7 50.0 50.0 35.2 50.0 38.7 71.1 26.2 50.0 19.5 68.1 26.2 85.1 50.0 68.1 50.0 77.8 68.1 35.2 6.4 7.7 22.2 68.1 50.0 18.2 16.0 83.6 50.0 15.1 50.0 50.0 22.2 26.2 96.9 50.0 50.0 50.0 50.0 53.9 53.9 41.2 50.0 53.9 50.0 34.8 50.0 85.0 61.8 50.0 32.0 50.0 78.6 63.9 16.0 3.3 30.0 23.5 63.7 50.0 64.8 71.0 16.5 64.8 50.0 60.2 8.7 50.0 35.2 50.0 24.5 70.1 50.0 68.1 53.9 31.0 50.0 72.1 50.0 64.8 93.0 50.0 81.9 89 35.2 64.8 53.9 81.9 53.9 50 72.1 50 68.1"
sent_prop = Prob(sentiment.split(' '))
sent_prop.create_freq_list()

v = SentimentIntensityAnalyzer()
# Get sentiment
score = v.polarity_scores(text)['compound'] #aggregated score -1 1 
score = (score * 50) + 50
score = round_up(score, 1)
print ("Sentence sentiment: " + str(score))
sent_prop.get_next(str(score))


#Subjectivity
subjectivity = "80.0, 0.0, 0.0, 0.0, 50.0, 68.8, 0.0, 0.0, 40.0, 0.0, 70.0, 100.0, 0.0, 0.0, 20.0, 0.0, 67.9, 53.6, 40.0, 40.0, 0.0, 0.0, 0.0, 0.0, 43.8, 30.1, 0.0, 0.0, 19.6, 0.0, 55.1, 0.0, 70.0, 63.0, 40.0, 80.0, 0.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 90.0, 0.0, 0.0, 47.5, 100.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 52.5, 0.0, 60.1, 0.0, 0.0, 0.0, 50.0, 50.0, 60.1, 66.2, 58.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 75.0, 0.0, 0.0, 0.0, 0.0, 80.0, 66.7, 40.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 95.0, 60.1, 0.0, 0.0, 0.0, 0.0, 66.7, 60.1, 84.4, 60.1, 0.0, 43.4, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 60.0, 0.0, 33.8, 0.0, 0.0, 0.0, 66.7, 0.0, 100.0, 0.0, 0.0, 67.5, 0.0, 100.0, 100.0, 50.0, 0.0, 0.0, 53.6, 0.0, 40.0, 100.0, 80.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 44.5, 0.0, 6.7, 53.6, 50.0, 0.0, 0.0, 0.0, 53.6, 0.0, 0.0, 0.0, 0.0, 50.0, 90.0, 0.0, 0.0, 0.0, 38.1, 0.0, 0.0, 68.8, 0.0, 0.0, 0.0, 100.0, 28.9, 0.0, 100.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.4, 0.0, 80.0, 60.0, 17.5, 50.0, 37.1, 0.0, 0.0, 0.0, 42.4, 0.0, 0.0, 0.0, 68.8, 0.0, 60.0, 50.0, 60.1, 30.0, 0.0, 30.0, 0.0, 0.0, 100.0, 40.0, 59.0, 84.8, 90.0, 0.0, 85.0, 20.0, 0.0, 30.0, 64.3, 60.1, 100.0, 0.0, 100.0, 65.0, 0.0, 0.0, 94.5, 0.0, 0.0, 75.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 88.9, 100.0, 63.0, 0.0, 68.0, 0.0, 53.5, 25.0, 53.6, 5.0, 42.5, 50.0, 55.1, 60.0, 0.0, 65.0, 40.0, 12.5, 68.8, 15.0, 15.0, 7.6, 20.0, 0.0, 0.0, 20.0, 0.0, 50.0, 0.0, 90.0, 0.0, 0.0, 0.0, 80.0, 100.0, 100.0, 0.0, 55.5, 0.0, 0.0, 0.0, 68.8, 90.0, 100.0, 5.0, 66.7, 0.0, 0.0, 88.9, 50.0, 0.0, 0.0, 0.0, 100.0, 40.0, 71.7, 90.0, 90.0, 75.0, 0.0, 0.0, 0.0, 0.0, 68.8, 100.0, 60.0, 88.9, 90.0, 0.0, 60.1, 0.0, 40.0, 40.2, 0.0, 0.0, 0.0, 20.0, 36.2, 56.7, 0.0, 0.0, 0.0, 100.0, 0.0, 48.4, 0.0, 0.0, 47.5, 0.0, 14.5, 100.0, 90.0, 70.0, 94.5, 0.0, 37.5, 0.0, 0.0, 0.0, 0.0, 0.0, 23.8, 0.0, 0.0, 40.0, 0.0, 0.0, 45.0, 93.4, 0.0, 58.8, 0.0, 0.0, 0.0, 0.0, 30.0, 35.0, 42.6, 60.1, 69.2, 44.6, 32.5, 77.6, 0.0, 80.0, 80.0, 100.0, 50.0, 30.1, 0.0, 0.0, 0.0, 0.0, 50.0, 54.2, 0.0, 0.0, 45.0, 60.1, 0.0, 59.1, 80.0, 0.0, 0.0, 0.0, 72.3, 0.0, 53.6, 0.0, 77.5, 50.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 83.4, 53.6, 30.0, 88.9, 0.0, 0.0, 0.0, 0.0, 36.0, 0.0, 43.4, 30.1, 30.1, 0.0, 100.0, 95.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 75.0, 0.0, 76.8, 0.0, 0.0, 0.0, 59.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 43.4, 0.0, 40.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 70.0, 0.0, 37.5, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 61.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 53.6, 0.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 53.6, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 86.7, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 53.6, 25.0, 51.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 66.7, 0.0, 0.0, 40.0, 100.0, 66.9, 66.7, 0.0, 81.3, 60.0, 64.7, 0.0, 36.0, 0.0, 34.4, 45.6, 100.0, 60.1, 50.0, 0.0, 0.0, 90.0, 0.0, 10.0, 100.0, 0.0, 0.0, 53.6, 66.7, 0.0, 37.1, 51.8, 0.0, 80.0, 0.0, 60.0, 62.5, 0.0, 62.5, 0.0, 53.6, 0.0, 0.0, 0.0, 31.7, 0.0, 0.0, 0.0, 30.0, 90.0, 0.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.0, 0.0, 0.0, 36.7, 100, 64.4, 55.1, 0.0, 100.0, 50.0, 60, 40, 0, 60.1, 0.0, 100.0"
subj_prop = Prob(subjectivity.split(", "))
subj_prop.create_freq_list()

subj = TextBlob(text).sentiment #0 objective, 1 subjective
subj = subj[1] * 100
subj = round_up(subj,1)
print ("Sentence subjectivity: " + str(subj))
subj_prop.get_next(str(subj))


#Verbs
verbs = "text show go keep know None None ’s come move mean ’s None gestures digs follow know None got ’s believe got None like follow decided use None take ’m None None None betcha get meet tell met ’s swear know likes come think tell ’s None None None smells talks need go None know gon keep None like belong ’re ’m mean look ’s go taken pass talkin None sayin see None None None None wanna call None talk ai ’m smack chill driving wants passed care None None gon None ai ’m None mean makin None None come None None know None None None gon None None None report know pulled ai ’s acts pulled ai rolling need got got None think tell None ’s say know kidding say None None None None believe ’m None recognize frustrates see stick place see ’s ’s None None say feel ’s pulled mean tell ’s ’s ’s know tell None cause took think got ’s None hurt know None None None None get ’s None None None None punch None None went throwed make know got drove went None None None None None snuff fainted used foolin None ’re think ai making think ’m make heard believe need need ai know find need None gon know None ca None ’s None None None comes None go None kick got beat None tell type None None None None ’s None None ca None ’s ’re think None None None telling squat drive runs go thought None None None wait keeping None pick feel need move move come stop get None talk play ’m play ’re None costing ’s care gon come suck None break call con talking make ’m saying saying ’s teach means ’s try see ’re move bend gon huff tire jump call practice know come works ’s ’s None showed go lying went imagine huff ai made gon None pick ’s mean asked need None None None None None come inherited ’s lose win ’m ’m go ’s hate made make ’s kidding None None None go like get know ’m saying ’m None talking None know go know gon None talking like believe telling asking hear None crush worry want understand ’re None understand ’m will None going know think develop None want ’m want shut know informs know None None gon need None love will None believe None find acting mean None noticed believe ’s happen misplaced None understand doing caught None went eyeing heard followed happened None walked None give None overhearing want ’s noticed seems None noticed went appears walking make caught notice think None heard sworn mean ’m None heard entered None None think made None shake tent know study None None None gone waddling utter None None think believe None witnessed waddling told None None mentioned mean having thought discussing love speak arrange find None None fired None could know covered found made know None None ’s None clear force ’s ’s None None None forgot ’m try seemed standing believe zoned made started wanted took kept found walking None ’s know ’s None None None get rest slleepp ’m coming rest hear rest ’m need ’s go come bother kidding told eating eat chew None breathe ’m breathe gon killed choking eat told need choke eating favor want ’s clean None took make get ’s invest adding doing ’s ’s enjoy None having know need saying like found ’m having wish ’s start ’m like says ’m hearing want get doing skipped want None jumping like go None None ’s None thought gon understood talked None care say care know None None trying trying trust want ’m live need take work happen run know happen ’s hear see see ’s understand stop take saying None know None get ’s None want playing None know None None get None ’s like like ’re None None need None leave understand want want ’s beat None None hurt None kicked saw ate come going want get hoping eat applied None live None feel know ’s work suck mean flip think need think nuke None ’s ’m find ’m mind coming loves love open None None None None None None None know None make None thinking ’s ’s ’s None breathing None choking ’s think attached ’s moving looks None think choking None None None fainted think tell breathing doing None getting ’s see None None None ’m think None mean None call spying save breathing None None ’s sitting None bouncing None record coming got None see ’s left None cheating think ’re ’re know know None None None None None ’s want ’ve ’s going want ’s ’s cheating ’s imagine None None picks calling None ’re answers know ’s answer None None going ’s borrow None look None going resting see calls tell want None call made ’re ’re calls ’re None draw ’s ’re close see None stop used None like go smell smell smell got smell make smell living go ’s got seem None go None watching warping ’m ’s ai start ’s None find seem mean None cross ’s mean ponder ai eat give wake work sleep ’re know ’m None None None ’m want gon take want wanna wish drown None want ’s None take told wanna cares want want want want want ’m want need got None take guess guess ’s got ai imagine want want give met mentioned knows None ’s forgive None None None None None ’s get got be None coming None None ’re None None ’s ’m swim want know go know whip see make give know think feel take come ’m ’s None feeling know feel None love mean will chill sitting working 's 'll  love"
verbs_prop = Prob(verbs.split(" "))
verbs_prop.create_freq_list()

#Extract verb form terminal
print (verbs_prop.get_next("s"))
#rint (verbs_prop.get_next("acting"))
print (verbs_prop.get_next("ll"))


def gen_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []
    input_text = seed_text
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len,truncating='pre')
        pred_word_ind = model.predict_classes(pad_encoded,verbose=0)[0]
        
        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += ' '+pred_word
        output_text.append(pred_word)
    return ' '.join(output_text)


model = load_model('./TextGen_Model.h5')
seq_len =  2
num_gen_words = 10
gen_text(model, tokenizer, seq_len, text, 10)
