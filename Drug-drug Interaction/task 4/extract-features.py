#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse

from deptree import *
import patterns


## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector

def extract_features(tree, entities, e1, e2) :


   feats = set() # get head token for each gold entity
   tkE1 = tree.get_fragment_head(entities[e1]['start'],entities[e1]['end'])
   tkE2 = tree.get_fragment_head(entities[e2]['start'],entities[e2]['end'])

   sentence_len = len(tree.get_nodes())
   if tkE1 is not None and tkE2 is not None:

      #Patterns from Session3
      p = patterns.check_wib(tree,tkE1,tkE2,entities,e1,e2)
      if p is not None: feats.add("type_wib=" + p)

      p = patterns.check_LCS_svo(tree,tkE1,tkE2)
      if p is not None: feats.add("type_LCS=" + p)

      p = patterns.check_wb(tree,tkE1,entities,e1)
      if p is not None: feats.add("type_wb=" + p)
      

      # features for tokens in between E1 and E2
      for tk in range(tkE1+1, tkE2) :
         if not tree.is_stopword(tk):
            word  = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            feats.add("lib=" + lemma)
            feats.add("wib=" + word)
            feats.add("lpib=" + lemma + "_" + tag)
            feats.add("tib="+tag)
            feats.add("lwib="+lemma+"_"+word)
            feats.add("wtib="+word+"_"+tag)
            feats.add("lwtib="+lemma+"_"+word+"_"+tag)
            
            # feature indicating the presence of an entity in between E1 and E2
            if tree.is_entity(tk, entities) :
               feats.add("eib")

      # features before E1
      for tk in range(tkE1):
         if not tree.is_stopword(tk):
            word  = tree.get_word(tk)
            tag = tree.get_tag(tk)
            lemma = tree.get_lemma(tk).lower()
            feats.add("libBefore=" + lemma)
            
            # feature indicating the presence of an entity before E1 
            if tree.is_entity(tk, entities) :
               feats.add("eibBefore")

      
      # features after E2
      for tk in range(tkE2,sentence_len):
         if not tree.is_stopword(tk):
            word  = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            feats.add("libAfter=" + lemma)
            feats.add("wibAfter=" + word)
            feats.add("ta="+tag)
            feats.add("lwa="+lemma+"_"+word)
            feats.add("wta="+word+"_"+tag)
            feats.add("lwta="+lemma+"_"+word+"_"+tag)
            # feature indicating the presence of an entity after E2
            if tree.is_entity(tk, entities) :
               feats.add("eibAfter")
      

      # features about paths in the tree
      lcs = tree.get_LCS(tkE1,tkE2)
      
      path1 = tree.get_up_path(tkE1,lcs)
      path1 = "<".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path1])
      feats.add("path1="+path1)

      path2 = tree.get_down_path(lcs,tkE2)
      path2 = ">".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path2])
      feats.add("path2="+path2)

      path = path1+"<"+tree.get_lemma(lcs)+"_"+tree.get_rel(lcs)+">"+path2      
      feats.add("path="+path)

      #features about edge labels
      feats.add("left_edge="+(path.split("<")[0]).split("_")[-1])
      feats.add("right_edge="+(path.split(">")[-1]).split("_")[-1])

      #features about entities labels
      feats.add("rel1="+tree.get_rel(tkE1))
      feats.add("rel2="+tree.get_rel(tkE2))

      #feature about the number of entities in the sentence
      feats.add("entities_number="+str(len(entities)))

      #features about the type of the entities
      feats.add("type1="+entities[e1]['type'])
      feats.add("type2="+entities[e2]['type'])
      
      # The following did not increase the result
      # But we did not comment it out for better readability

      # check if path connecting e1 to e2 (over lcs) has third entity
      pathE1 = tree.get_up_path(tkE1,lcs)
      pathE2 = tree.get_up_path(tkE2,lcs)
      pathE1_E2 = pathE1 + pathE2 + [lcs]
      # remove tkE1 and tkE2 from path as we want to look for a third  entity
      if tkE1 != tkE2: 
        pathE1_E2.remove(tkE1)
        pathE1_E2.remove(tkE2)
        for elem in pathE1_E2: 
            if tree.is_entity(elem,entities):
                feats.add("ThirdEntity_in_path")
                break

      # features about lcs (PoS, lemma..)
      lcs = tree.get_LCS(tkE1,tkE2)
      feats.add("lcsPoS="+tree.get_tag(lcs))
      feats.add("lcsLemma="+tree.get_lemma(lcs))
      feats.add("lcsWord="+tree.get_word(lcs))
   return feats


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
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
        stext = s.attributes["text"].value   # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")           
           entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1 : continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi=="true") : dditype = p.attributes["type"].value
            else : dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction

            feats = extract_features(analysis,entities,id_e1,id_e2) 
            # resulting vector
            print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")

