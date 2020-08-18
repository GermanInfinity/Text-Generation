import requests 
import urllib.request
import time, types
from bs4 import BeautifulSoup
import csv, sys

CSV_name = "30scenes.csv"

sources = ["https://monologueblogger.com/ill-be-your-tour-guide/",
			"https://monologueblogger.com/bad-talk/",
			"https://monologueblogger.com/fear-of-the-fat-man/",
			"https://monologueblogger.com/life-moves/",
			"https://monologueblogger.com/the-duck/",
			"https://monologueblogger.com/weight-of-laughter/",
			"https://monologueblogger.com/smile-for-the-camera/",
			"https://monologueblogger.com/tea-time/",
			"https://monologueblogger.com/boulevard/",
			"https://monologueblogger.com/the-handler/",
			"https://monologueblogger.com/fix/",
			"https://monologueblogger.com/shribble/",
			"https://monologueblogger.com/story-plucked-lobster/",
			"https://monologueblogger.com/mickeys-cake/",
			"https://monologueblogger.com/imaginary-you/",
			"https://monologueblogger.com/way-of-the-wiffle-ball-bat/",
			"https://monologueblogger.com/window-pain/",
			"https://monologueblogger.com/last-wednesday/",
			"https://monologueblogger.com/two-three-days/",
			"https://monologueblogger.com/entirely-as-well/",
			"https://monologueblogger.com/not-a-care-in-the-world/",
			"https://monologueblogger.com/less-soap-more-blood/",
			"https://monologueblogger.com/chicken-cutlet-sandwich-and-some-wings/",
			"https://monologueblogger.com/odds-fifty-fifty/",
			"https://monologueblogger.com/jump-off-earth/",
			"https://monologueblogger.com/window-pain/"]


""" Extract text from sources """	
master_list = []
for source in sources:
	response = requests.get(source)
	soup = BeautifulSoup(response.text, 'html.parser')

	"""Finding html content"""
	container = soup.find('main', attrs={'class':'clearfix'})
	content = []

	done = False
	count = -1

	for node in soup.findAll('p'):
		if done == False:
			a = [count]
			query = node.findAll(text=True)
			if len(query) == 2: 
				store = ''.join(query)
				a.append(store)
				content.append(a)

			if query == 'CREATE': break


	content = content[2:]
	master_list += content


""" Fix numbering """ 
count = 1
for ele in master_list: 
	ele[0] = count
	count += 1



""" Put master_list in a csv file """ 
with open(CSV_name, "w", newline="") as f:
	writer = csv.writer(f)
	writer.writerows(master_list)
