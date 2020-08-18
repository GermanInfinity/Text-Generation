from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from pickle import dump,load
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model
import math


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

n_sequences = np.empty([len(sequences),train_len], dtype='int32')
for i in range(len(sequences)):
    n_sequences[i] = sequences[i]


train_inputs = n_sequences[:,:-1]
train_targets = n_sequences[:,-1]

train_targets = to_categorical(train_targets, num_classes=vocabulary_size+1)
seq_len = train_inputs.shape[1]
train_inputs.shape



def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len,input_length=seq_len))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(vocabulary_size,activation='softmax'))
    opt_adam = keras.optimizers.Adam(lr=0.001) 
    #You can simply pass 'adam' to optimizer in compile method. Default learning rate 0.001
    #But here we are using adam optimzer from optimizer class to change the LR.
    model.compile(loss='categorical_crossentropy',optimizer=opt_adam,metrics=['accuracy'])
    model.summary()
    return model


from tensorflow import keras
model = create_model(vocabulary_size+1,seq_len)
path = './checkpoints/word_pred_Model4.h5'
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit(train_inputs,train_targets,batch_size=128,epochs=1250,verbose=1,callbacks=[checkpoint])
#model.save('./TextGen_Model.h5')
dump(tokenizer,open('tokenizer_Model4','wb'))   
model = create_model(vocabulary_size+1,seq_len)
path = './checkpoints/word_pred_Model4.h5'
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit(train_inputs,train_targets,batch_size=128,epochs=1250,verbose=1,callbacks=[checkpoint])
#model.save('./TextGen_Model.h5')
dump(tokenizer,open('tokenizer_Model4','wb'))   



from keras.preprocessing.sequence import pad_sequences
model = model
tokenizer = load(open('./tokenizer_Model4','rb'))
seq_len = 3
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




### CREATE A LIST OF LISTS FOR 10-FOLDS: VALIDATION DATA
### TESTING PURPOSES
text = "We the excrement that’s been pooped out from some giant ..." 
text = text.split(" ")
print (len(text))

res = []
ans = ""
count = 0
for ele in text: 
    ans += ele
    ans += " "
    if count == 59: 
        count = 0
        res.append(ans)
        ans = ""
    count += 1

print (res)
text = res
print (text)
# for ele in res: 
#     print (len(ele.split(" ")))



count = 1
for fold in text: 
    print ("Fold " + str(count))
    count += 1

    sol = []
    sol2 = []


    test = fold.split(' ')
    string = ""
    for ele in test: 
        string += ele
        string += " "

        encoded_text = tokenizer.texts_to_sequences([ele])[0]
        encoded_text2 = tokenizer.texts_to_sequences([string])[0]

        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len,truncating='pre')
        pad_encoded2 = pad_sequences([encoded_text2], maxlen=seq_len,truncating='pre')
    
        prob = model.predict_proba(pad_encoded, verbose=0)
        prob2 = model.predict_proba(pad_encoded2, verbose=0)

        sol.append(round(np.amax(prob),4))
        sol2.append(round(np.amax(prob2),4))


    t = len(sol)
    res = 1
    for ele in sol: 
        res = res * (ele)
        
    res = float(pow(res,1/t))
    res = 1/res
    print ("Sol:" + str(res))

    t = len(sol2)
    res = 1
    for ele in sol2: 
        res = res * (ele)
    res = float(pow(res,1/t))
    res = 1/res
    print ("Sol2:" + str(res))

