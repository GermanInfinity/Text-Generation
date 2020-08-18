import csv, pickle 

# file = open('./SLA.csv', 'r')
# reader = csv.reader(file)
# all_sent = []
# all_subj = []
# for row in reader: 
#     sentiment = row[0]
#     subj = row[1]
#     all_sent.append(sentiment)
#     all_subj.append(subj)


# with open('sentiment.txt', 'w') as output: 
#     output.write(str(all_sent))

# with open('subjectivity.txt', 'w') as output: 
#     output.write(str(all_subj))



a = open("./sentiment.txt")
sent = []
lines = a.readlines()
for line in lines: 
	sent.append(line.replace(',',''))


# with open('sentiment.txt', 'w') as output: 
#     output.write(str(sent))

# a.close()

ret = ""
for ele in line:
	if ele == "'": continue
	ret += ele

ret = ret.split(" ")

#Picklize sla data 
with open('./sentiment-data.pkl', 'wb') as f:
 	pickle.dump(ret, f)

