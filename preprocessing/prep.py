import pandas as pd 
import csv 
import math
import pickle 
#df = pd.read_csv('./data.txt')
#data = df['text'].str.cat(sep=' ')
#a = df.split(' ')

# with open('data.txt', 'w') as f:
# 	for item in a: 
# 		f.write("%s " % item)


file = open('./SLA.csv', 'r')
# # #a = file.read()
# # #a = a.split(' ')

reader = csv.reader(file)
new_list = []
for row in reader:
	b = float(row[0])
	c = float(row[1])

	new_list.append(b)
	new_list.append(c)

	b = b * b
	c = c * c
	a = b + c

	a = pow(a, 1/2)
	a = round(a,3)
	new_list.append(a)

#print (len(new_list))

### TRAIN AND TEST SET SPLIT ###
train = new_list[:1473]
test = new_list[1473:]


print (test)
#Picklize train and test set 
with open('./train.pkl', 'wb') as f:
 	pickle.dump(train, f)

with open('./test.pkl', 'wb') as f:
 	pickle.dump(test, f)

print ("DONE")


with open('./train.pkl', 'rb') as f:
	# Store in list
	LIST = pickle.load(f)

