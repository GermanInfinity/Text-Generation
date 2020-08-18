import csv
data = open('30scenes.csv', 'r')
reader = csv.reader(data)

new_csv = "30scenes-edited.csv"
new_list = []

""" Sanitize content into a new list """
for row in reader: 
    container = [row[0]]
    cut_index = row[1].index(" ")
    details = row[1][cut_index+1:]
    container.append(details)
    new_list.append(container)

""" Write into new CSV """
with open(new_csv, "w", newline="") as f: 
    writer = csv.writer(f)
    writer.writerows(new_list)


"""             Auxilliary functions                """
### Extractes sentences from a block of text, sentences serparated by a full
### stop. 
def extract_sentences(self, paragraph): 
    return paragraph.split('.')