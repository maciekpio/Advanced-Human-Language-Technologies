#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize


   
## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens):
    
       # load external databases HSDB and DrugBank
       external = {}
       with open("../session1/HSDB.txt") as h :
        for x in h.readlines() :
            external[x.strip().lower()] = "drug"
       with open("../session1/DrugBank.txt",encoding="utf-8") as h :
        for x in h.readlines() :
            (n,t) = x.strip().lower().split("|")
            external[n] = t
    
       # add some helper functions
       # check and return True if text has any numbers
       def hasNumbers(tokenForm):
           return any(elem.isdigit() for elem in tokenForm)
    
       # check and return True if text has any special characters 
       def hasSpecialCharacter(tokenForm):
           return any(not elem.isalnum() for elem in tokenForm)
    
       # for each token, generate list of features and add it to the result
       result = []
       for k in range(0,len(tokens)):
          tokenFeatures = [];
          t = tokens[k][0]
    
          # add the text as feature
          tokenFeatures.append("form="+t)
          # add a suffix of last three and last five characters as features
          tokenFeatures.append("suf3="+t[-3:])
          tokenFeatures.append("suf5="+t[-5:])
    
          # add feature if text is written in capitalized letters
          if t.isupper() : tokenFeatures.append("isupper")
    
          # if text has numbers/special characters, add a new feature
          if hasNumbers(t) : tokenFeatures.append("hasNumbers")
          if hasSpecialCharacter(t) : tokenFeatures.append("hasSpecialCharacter")
    
          # check same databases as in session1 and add a feature if we have a match
          if t.lower() in external : tokenFeatures.append(str(external[t.lower()]))
    
          # do the same things for next and previous words
          if k>0 :
             tPrev = tokens[k-1][0]
             tokenFeatures.append("formPrev="+tPrev)
             tokenFeatures.append("suf3Prev="+tPrev[-3:])
             tokenFeatures.append("suf5Prev="+tPrev[-5:])
             if tPrev.isupper() : tokenFeatures.append("isupperPrev")
             if hasNumbers(tPrev) : tokenFeatures.append("hasNumbersPrev")
             if hasSpecialCharacter(tPrev) : tokenFeatures.append("hasSpecialCharacterPrev")
             if tPrev.lower() in external : tokenFeatures.append(str(external[tPrev.lower()]))
          else :
             tokenFeatures.append("BoS")
    
          if k<len(tokens)-1 :
             tNext = tokens[k+1][0]
             tokenFeatures.append("formNext="+tNext)
             tokenFeatures.append("suf3Next="+tNext[-3:])
          else:
             tokenFeatures.append("EoS")
        
          result.append(tokenFeatures)
        
       return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
