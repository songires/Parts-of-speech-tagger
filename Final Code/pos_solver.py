###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids: ssongire Suraj zhouco Cong
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
'''
Report:

Aim: Given a test sentence we are suppose to find out the parts of speech for every word in that sentence with the help of knowledge of training data 
provided to us.


Initially, the training data is used to gather the knowledge about various probabilities of occurrences of words and parts of speech.
In the training function we compute and store 6 different dictionaries.

DATASTRUCTURE:

1. Wordcount{} : This dictionary stores the total count for every word occured in training data as value of that corresponding word.

2. Partcount{} : This dictionary stores the total count of every part of speech occurring in the training data.

3. Part_start_probability{} : This dictionary stores the probability of every part of speech occurring at the start of the sentence.
   It can also be denoted as P(S1). which is the probability of first part of speech.

4. Part{} : This dictionary is created for a handy use. Although it doesn't have much work in the code but it is used to reference the part of speech
   present in the training set. In this dictionary we have given every word a unique number for reference.

5. WordSpeech{} : WordSpeech dictionary is a dictionary which stores the value of probability of number of times the particular word occurs in a 
   particular part of speech. This can also be denoted as P(Wi|Si). Every key in this dictionary is referenced as---> Word + '_' + Part of Speech  .

6. nextspeech{} : This dictionary stores the probability of every part of speech occurring next to every other part of speech. It can also be denoted as
P(Si+1|Si).


PROCEDURE:

1. NAIVE METHOD: 
For detecting the part of speech for a given sentence. We run a loop for every word in sentence. We check the probability of every word occuring with every 
part of speech which can be denoted as P(Wi|Si). This can be obtained from the word speech dictionary. Now we multiply this value with the probability of 
part of speech occurring in the training set. Now we choose the maximum value of this multiplication.
For whichever part of speech this value is maximum we choose that corresponding part of speech for that particular word.

2. MCMC sampling:
In this sampling method we generate 5 samples for the sentence to be tested. These samples are generated using Gibb's Sampling method. Gibb's sampling is a special case
of MCMC sampling algorithm. This step is just to generate the 5 samples for testing if the sampling works. Although the accuracy is less, we have just tested it for 5 samples 
but as the number of samples increamse the accuracy will eventually increase. 
For this step we take entire sentence and initially assign a random tag to all the words. Tag here means the part of speech. Now, for every sample we we randomly select one word
and its tag. Now for that position in the sentence we have 2 cases:
	1. the word is first word :	If the word is the first word, we calculate the probability of that word with every part of speech. We get the walue with the formula
					P(S1|S2S3....Sn, W1W2...Wn) = P(W1|S1)P(S2|S1)
					Thus the above probability  refers ot the possibility of particular part of speech given all other part of speech and all the words,
					and that can be calculated as probability of that word given part of speech which we have already calculated in word speech dictionary,
					multiplied by probability of next part of speech given in the previous sample.
	2. The word is any other position word :
					If the word is not the first word, we calculate the probability of that word with every part of speech. We get the walue with the formula
					P(Si|S2S3....Sn, W1W2...Wn) = P(Wi|Si)P(Si+1|Si)P(Si-1|Si)
					Thus the above probability  refers ot the possibility of particular part of speech given all other part of speech and all the words,
					and that can be calculated as probability of that word given part of speech which we have already calculated in word speech dictionary,
					multiplied by probability of next part of speech given in the previous sample multiplied by the probability of current part of speech
				        given previous part of speech
After calculating this value for every part of speech for a particular word we assign tag which has highest value.
Note that for this step we only generate the next sample, i.e. we only change the tag of that partiular position and keep rest of the tags same. So next sample has only one value changed.

Finally we return all the samples back.


3. Max Marginal:
In this method we use the above MCMC sampling method to generate N number of samples. Samples are generated in similar manner as stated above. But there are 2 more constraints which i have added to get good result. 
Firstly, I have ignored first few samples inorder to get started, this is also called the warm up period.
Second is we keep track of every samples. 
For calculating part of speech for a particular word we follow these steps:
	a) We calculate the probability of all the parts of speech of that position of the sentence, i.e we calculate how many time each part of speech appear at that position in the 		   test sentence. Whichever has highest probability of appearing we assume that, that part of speech is correct for that position and store its probability and name in the list. 		   for each tag. 

4. Vitterbi Algorithm:
in this function we use Vitterbi algorithm to backtrack the maximum possible sequene of part of speechs [s1, s2, ..., sn] appearing in the given sentence.
In this method again there are 2 cases:
	a) when the word is a start sentence: When its a start sentence we use the start probability of speech and the emmission probability of that speech of that particular word to 		   calcute the value and store it in a dictionary.
	b) when the word is not in start of sentence then for each possible part of speech we calculate the value and store it in a different dictionary.
	   this value is calculated using formula for vitterbi algorithm. 
	   Emmission(W|Si) * argmax(Transition Probability of Si-1 to Si * Previous stae probability)
	   In the dictionary I also store the state from which the maximum value is generated so that while we backtrack we can directly look up on the state we have saved.
	c) Then, we store all the values generated in the dictionary into a list.
	d) Finally we backtrack the list and generate all the tags which we have traversed.


Best algorithm:
Here we have applied combination of Vitterbi and Naive algorithm to calculate the best results for the sentence part of speech prediction.

'''
####

from random import randint
import math
import random
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

wordcount = {}
partcount = {}
part_start_probability = {}     
part = {}   #Pos unique key is given to each
wordspeech = {} # pwi|Si  Key = word+'_"+partof speech name
nextspeech = {} #pSi+1|Si
class Solver:

    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        
        term1 = 0
        
        value2 = math.log(part_start_probability[label[0]]/10000.0)        
        
        for i in range(0,len(sentence)):
            word = sentence[i]
            speechi = label[i]
            key1 = word+"_"+speechi
            if key1 in wordspeech:
                value = wordspeech[key1]/10000.0
                temp1 = math.log(value)
                term1 = term1+temp1
            else:
                term1 = term1+0.0
        for i in range(1,len(sentence)-1):
            speechi = label[i]
            speechnexti = label[i+1]
            key2 = speechi+"_"+speechnexti
            if key2 in nextspeech:
                temp2 = math.log(nextspeech[key2])
                value2 = value2 + temp2
            
        newterm = value2 + term1
        
        return newterm

    # Do the training!
    #
    def train(self, data):
        count = 0
        for obj in data:
            for pos in obj[1]:
                if pos == obj[1][0]:        
                    if pos in part_start_probability:                   #partof speech start count
                        part_start_probability[pos] += 1
                    else:
                        part_start_probability[pos] = 1
                                                

                if pos in partcount:                                    #partofspeech total count
                    partcount[pos] += 1
                else:
                    part[pos] = count
                    partcount[pos] = 1
                    count += 1
                    
            for word in obj[0]:                                         #total wrd count
                if word in wordcount:
                    wordcount[word] += 1
                else:
                    wordcount[word] = 1

            for counter in range(0,len(obj[0])):
                word = obj[0][counter]
                pos = obj [1][counter]
                key = word+"_"+pos
                if key in wordspeech:
                    wordspeech[key] += 1
                else:
                    wordspeech[key] = 1

            for counter in range(1,len(obj[0])):
                prev = obj[1][counter-1]
                nxt = obj[1][counter]
                key = prev+"_"+nxt
                if key in nextspeech:
                    nextspeech[key] +=1
                else:
                    nextspeech[key] = 1
        
        for key in nextspeech:
            splitlist = key.split("_")
            nextspeech[key] = round(nextspeech[key]/float(partcount[splitlist[1]]),4)
        

        
        for key in wordspeech:                                      #P[Wi|Si]
            wordspeechlist = key.split("_")
            wordspeech[key] = round(((wordspeech[key]/float(partcount[wordspeechlist[1]])*10000) + 0.0001),4)
        
        
        for pos in part_start_probability:
            part_start_probability[pos] = round((part_start_probability[pos]/float(len(data))*10000)+0.0001,4) #P[Si]
            
            
        for pos in part:
            if pos not in part_start_probability:
                part_start_probability[pos] = 0.0001
                
        pass
        
    # Functions for each algorithm.
    #
    def naive(self, sentence):
               
        tags = []
        
        for word in sentence:
            maxvalue = 0
            for pos in part:
                key = word+"_"+pos
                value = wordspeech.get(key,0)*partcount[pos]
                if value>maxvalue:
                    tag = pos
                    maxvalue = value
                
            if(maxvalue == 0):
                tag = 'noun'
            tags.append(tag)
        return [ [tags], [] ]        
        
        
    def mcmc(self, sentence, sample_count):
        samples = []
        naive = ["noun"]*len(sentence)
        forcopy = naive[:]
        slist = []
        actual = sample_count
        sample_count += 100
        for i in range(0,sample_count):
		if i> (sample_count-actual):
			forcopy = naive[:]
                	samples.append(forcopy)
		
                if len(sentence)==1:
                        samples.append([ "." ] * len(sentence))
                elif len(sentence) >1:
                        n = randint(1,len(sentence))
                        del slist[:]
                        if n == 1:
                            w1 = sentence[0]
                            for s1 in part:
                                key1 = w1+"_"+s1
                                s2 = naive[1]
                                key2 = s1+"_"+s2
                                if key1 in wordspeech and key2 in nextspeech:
                                    value = wordspeech[key1]*nextspeech[key2]
                                else:
                                    value = 0.0001
                                slist.append([value,s1])
                            maxvalue = -1
                            for obj in slist:
                                if maxvalue<obj[0]:
                                    maxvalue = obj[0]
                                    tag = obj[1]
                            naive[0]=tag
                        elif n > 1 and n < len(sentence):
                            wi = sentence[n-1]
                            for si in part:
                                key1 = wi+"_"+si
                                siplus1=naive[n]
                                si_1=naive[n-2]
                                key2=si+"_"+siplus1
                                key3=si_1+"_"+si
                                if key1 in wordspeech and key2 in nextspeech and key3 in nextspeech:
                                    value = wordspeech[key1]*nextspeech[key2]*nextspeech[key3]
                                else:
                                    value = 0.0001
                                slist.append([value,si])
                            maxvalue = -1
                            for obj in slist:
                                if maxvalue<obj[0]:
                                    maxvalue = obj[0]
                                    tag = obj[1]
                            naive[n-1]=tag
                        elif n == len(sentence):
                            wn = sentence[n-1]
                            for sn in part:
                                key1 = wn+"_"+sn
                                sn_1=naive[n-2]
                                key3=sn_1+"_"+sn
                                if key1 in wordspeech and key3 in nextspeech:
                                    value = wordspeech[key1]*nextspeech[key3]
                                else:
                                    value = 0.0001
                                slist.append([value,sn])
                            maxvalue = -1
                            for obj in slist:
                                if maxvalue<obj[0]:
                                    maxvalue = obj[0]
                                    tag = obj[1]
                            naive[n-1]=tag
		
        return [ samples, [] ]
 
    def max_marginal(self, sentence):        
        samples = [] 
        naive = ["noun"]*len(sentence)
        slist = []
        sample_count=1000
        for i in range(0,sample_count):
            if len(sentence)==1:
                flag = 0
                word = sentence[0]
                maxval = -1
                for speech in part:
                    keynext = word+"_"+speech
                    if(keynext in wordspeech):
                        if(maxval<wordspeech[keynext]):
                            maxval = wordspeech[keynext]
                            t = speech
                            flag = 1
                        else:
                            pass
                if flag == 1:
                    samples.append([t])
                else:
                    samples.append(['noun'])
                    

            elif len(sentence) >1:
                forcopy = naive[:]
                samples.append(forcopy)
                n = randint(1,len(sentence))
                del slist[:]
                if n == 1:
                    w1 = sentence[0]
                    for s1 in part:
                        key1 = w1+"_"+s1
                        s2 = naive[1]
                        key2 = s1+"_"+s2
                        if key1 in wordspeech and key2 in nextspeech:
                            value = wordspeech[key1]*nextspeech[key2]
                        else:
                            value = 0.0001
                        slist.append([value,s1])
                    maxvalue = -1
                    for obj in slist:
                        if maxvalue<obj[0]:
                            maxvalue = obj[0]
                            tag = obj[1]
                    naive[0]=tag
                elif n > 1 and n < len(sentence):
                    wi = sentence[n-1]
                    for si in part:
                        key1 = wi+"_"+si
                        siplus1=naive[n]
                        si_1=naive[n-2]
                        key2=si+"_"+siplus1
                        key3=si_1+"_"+si
                        if key1 in wordspeech and key2 in nextspeech and key3 in nextspeech:
                            value = wordspeech[key1]*nextspeech[key2]*nextspeech[key3]
                        else:
                            value = 0.0001
                        slist.append([value,si])
                    maxvalue = -1
                    for obj in slist:
                        if maxvalue<obj[0]:
                            maxvalue = obj[0]
                            tag = obj[1]
                    naive[n-1]=tag
                elif n == len(sentence):
                    wn = sentence[n-1]
                    for sn in part:
                        key1 = wn+"_"+sn
                        sn_1=naive[n-2]
                        key3=sn_1+"_"+sn
                        if key1 in wordspeech and key3 in nextspeech:
                            value = wordspeech[key1]*nextspeech[key3]
                        else:
                            value = 0.0001
                        slist.append([value,sn])
                    maxvalue = -1
                    for obj in slist:
                        if maxvalue<obj[0]:
                            maxvalue = obj[0]
                            tag = obj[1]
                    naive[n-1]=tag
		
        Si_count={}
        tag=[]
        probv=[]
        for i in range(0, len(sentence)):
                Si_count.clear()
                maxvalue = -1
                for pos in part:
                    for j in range(0, len(samples)):
                        if pos == samples[j][i]:
                            if pos in Si_count:
                                Si_count[pos]+=1
                            else:
                                Si_count[pos]=1
                    
                    if pos in Si_count and Si_count[pos]>maxvalue:
                         maxvalue=Si_count[pos]
                         finalpos=pos
                    else:
                        pass

                tag.append(finalpos)
                probv.append(round(maxvalue/float(sample_count),4))
        return [[ tag], [probv,]]
                    
                                  
    def best(self, sentence):
	tags = []
        
        for word in sentence:
            maxvalue = 0
            for pos in part:
                key = word+"_"+pos
                value = wordspeech.get(key,0)*partcount[pos]
                if value>maxvalue:
                    tag = pos
                    maxvalue = value
                
            if(maxvalue == 0):
                tag = 'noun'
            tags.append(tag)
        return [ [tags], [] ] 
#-----------------------

    def viterbi(self, sentence):
        tags = []
        prevdict = {}
        temp = {}
        l = []
        count = 0
        for word in sentence:
            maxvalue = 0.0
            for pos in part:
                key = word+"_"+pos
                if count == 0:
                    if pos in part_start_probability and key in wordspeech:   
                            prevdict[pos] = [round(part_start_probability[pos]*wordspeech[key]*100,4),pos]
                            #print prevdict
                    else:
                        prevdict[pos] = [part_start_probability[pos]*1000,pos]
                            
                else:
                     
                         
                     for speech in prevdict:
                            key1 = speech+"_"+pos
                            if key1 in nextspeech:           #check else do 0 wali step
                                variable = round((prevdict[speech][0]*nextspeech[key1]*100),4)
                                if variable > maxvalue:
                                    maxvalue = variable
                                    lasttag = speech
                     
                     if key in wordspeech: 
                         temp[pos] = [round(maxvalue*wordspeech[key],4),lasttag]
                        
                     else:
                        temp[pos] = [round(maxvalue*0.0001,4),lasttag]
                     maxvalue = 0
            if(count!=0):
                prevdict = temp.copy()
                temp.clear()                          
            l.append(prevdict)
            count += 1  
        
        maxval = -1
        
        l = l[::-1]
        d = l[0]
        
        for key in d:
            if maxval<d[key][0]:
                maxval = d[key][0]
                nextkey = d[key][1]
                currentkey = key
        tags.append(currentkey)

        for d in range(1,len(l)):
            dictionary = l[d]
            tags.append(nextkey)
            nextkey = dictionary[nextkey][1]
        
        tags = tags[::-1]
        
        
        return [ [ tags], [[0] * len(sentence),] ]



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"

