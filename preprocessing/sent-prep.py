a = open("./sentiment.txt")
lines = a.readlines()


ret = ""
for ele in lines[0]:
	if ele == "'": continue
	ret += ele
print (type(ret))
#print (ret[0])


