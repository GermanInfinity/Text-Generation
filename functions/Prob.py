import random
import numpy as np 
class Prob:
    # Class takes in a list of elements
    def __init__(self, sequence):
        self.sequence = sequence
        self.prob_dict = {}
        self.length = len(self.sequence)
        self.min_dist = []

    # for every token in your corpus, we have a probability of the next token 
    # given the history
    def create_freq_list(self):
        unique_list = []
        for ele in self.sequence: 
            if ele not in unique_list: unique_list.append(ele)

        prob_dict = {}

        #Get frequencies
        for ele in unique_list:
            prob_dict[ele] = self.sequence.count(ele)
        #Determine probabilities
        for ele in prob_dict:
            prob_dict[ele] = prob_dict[ele] / self.length

        self.prob_dict = prob_dict
        return (prob_dict)

    # Function updates probability distribution dictionary
    def update_freq_list(self, new_element):
        self.sequence.append(new_element)
        self.length = len(self.sequence)
        self.prob_dict = self.create_freq_list()
        return self.prob_dict

    
    def get_dist_seq(self, history):
        #get prob of each element in history
        prob_history = 1
        for ele in history: 
            if ele not in self.prob_dict: 
                self.update_freq_list(ele)

        for ele in history: 
            if ele in self.prob_dict:
                prob_history *= self.prob_dict[ele]
               
        #check all elements prob
        elem_probs = {}
        for ele in self.prob_dict:
            elem_probs[ele] = self.prob_dict[ele] * prob_history


        #get occurences of elements in history 
        hist_index = []
        for idx in range(len(history)): 
            hist_small = []
            for idx2 in range(len(self.sequence)): 
                if history[idx] == self.sequence[idx2]:
                    hist_small.append(idx2)
            hist_index.append(hist_small)

        #make lists equal sized
        len_of_hist = []
        for ele in hist_index:
            len_of_hist.append(len(ele))
        size = max(len_of_hist)

        #make history elements same size
        for ele in hist_index:
            if len(ele) != size:
                space = size - len(ele)
                num = random.choice(ele)

                for idx in range(space): ele.append(num)


        #get distance between history elements
        ans = []
        for idx in range(len(hist_index) - 1): #for loop for outer lists
            small_diff = []
            for idxx in range(len(hist_index[idx])):
                for idxy in range(len(hist_index[idx])):
                    small_diff.append(abs(hist_index[idx+1][idxx] - hist_index[idx][idxy]))

            ans.append(small_diff)

        #get minimum distance between history elements
        for ele in ans: 
            self.min_dist.append(min(ele))

        #print (min_dist)
        #return self.min_dist
    
        #

    # Given some history, produce the next viable output 
    def get_next_word(self, history):
        self.get_dist_seq(history)
        #prob of each history word
        hist_prob = 1
        for ele in history: 
            hist_prob *= self.prob_dict[ele]

        #asymptotic difference
        ln_dist = 1
        for ele in self.min_dist: 
            ln_dist *= 1/ele 

        dict_prob = {}
        for ele in self.prob_dict: 
            dict_prob[ele] = self.prob_dict[ele] * ln_dist

        #print (dict_prob)
        return max(dict_prob , key=self.dict_prob.get)



## Testing create_freq_list
seq = 'what happened today was really terrific i and i hope it happens again anyway what you are going through is not forever. terrific person i hope good things happen to everyone'
test = Prob(seq.split(' '))
test.create_freq_list()


## Testing get_next
seq2 = 'apex i hope so to'
test.get_next_word(seq2.split(' '))
