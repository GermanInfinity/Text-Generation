from DPR import DPR
import nltk
import spacy
import csv



if __name__ == "__main__":

    data = open('../RNN/30scenes-edited.csv', 'r')
    reader = csv.reader(data)

    new_csv = "verbs.csv"

    DPR = DPR("en_core_web_md")

    stored_verbs = []
    count = 1


    for row in reader:
        if count == 1: 
            count += 1
            continue


        verbs = DPR.get_verb(row[1])

        if verbs is None: 
            store = []
            store.append('None')
            stored_verbs.append(store)
            continue

        for verb in verbs: 
            store = []
            store.append(str(verb))
            stored_verbs.append(store)

 

   


    with open(new_csv, "w", newline="") as f: 
        writer = csv.writer(f)
        writer.writerows(stored_verbs)