## ------------------- 
## -- check pattern:  LCS is a verb, one entity is under its "nsubj" and the other under its "obj"      

def check_LCS_svo(tree,tkE1,tkE2):

   if tkE1 is not None and tkE2 is not None:
      lcs = tree.get_LCS(tkE1,tkE2)

      if tree.get_tag(lcs)[0:2] == "VB" :      
         path1 = tree.get_up_path(tkE1,lcs)
         path2 = tree.get_up_path(tkE2,lcs)
         func1 = tree.get_rel(path1[-1]) if path1 else None
         func2 = tree.get_rel(path2[-1]) if path2 else None
         
         if (func1=='nsubj' and func2=='obj') or (func1=='obj' and func2=='nsubj') :
            lemma = tree.get_lemma(lcs).lower()
            if lemma in ['diminish','augment','experience','counteract','potentiate',
                        'enhance','reduce','antagonize','include',
                        'lower'] :
               return 'effect'
            if lemma in ['impair','inhibit','displace','accelerate','bind','induce',
                        'decrease','elevate','delay','produce',
                        'prolong','cause','show'] :
               return 'mechanism'
            if lemma in ['exceed','should'] :
               return 'advise'
            if lemma in ['suggest'] :
               return 'int'
         
   return None

## ------------------- 
## -- check pattern:  A word in between both entities belongs to certain list

def check_wib(tree,tkE1,tkE2,entities,e1,e2):

   if tkE1 is not None and tkE2 is not None:
      # get actual start/end of both entities
      l1,r1 = entities[e1]['start'],entities[e1]['end']
      l2,r2 = entities[e2]['start'],entities[e2]['end']
      p = []
      for t in range(tkE1+1,tkE2) :
         # get token span
         l,r = tree.get_offset_span(t)
         # if the token is in between both entities
         if r1 < l and r < l2:
            lemma = tree.get_lemma(t).lower()
            if lemma in ['tendency','stimulate','regulate','prostate',
            'modification','augment','accentuate','exacerbate'] :
               return 'effect'
            if lemma in ['react','metabolism','faster','presumably','induction',
            'substantially','minimally','concentration']:
               return 'mechanism'
            if lemma in ['exceed','extreme','cautiously','should']:
               return 'advise'
            if lemma in ['interact'] :
               return 'int'

   return None

## ------------------- 
## -- check pattern:  A word before first entity belongs to certain list

def check_wb(tree,tkE1,entities,e1,):

   if tkE1 is not None :
      # get actual start/end of the first entity
      l1,r1 = entities[e1]['start'],entities[e1]['end']
      p = []
      for t in range(tkE1) :
         # get token span
         l,r = tree.get_offset_span(t)
         # if the token is in before the entity
         if l1 > r:
            lemma = tree.get_lemma(t).lower()
            if lemma in ['should','caution']:
               return 'advise'

   return None

# The following did not increase the result

## -- check pattern:  A word after second entity belongs to certain list
def check_wa(tree,tkE1,tkE2,entities,e1,e2):

   if tkE1 is not None and tkE2 is not None:
      # get actual start/end of both entities
      l1,r1 = entities[e1]['start'],entities[e1]['end']
      l2,r2 = entities[e2]['start'],entities[e2]['end']
      p = []
      for t in range(tkE2,tree.get_n_nodes()):
         # get token span
         l,r = tree.get_offset_span(t)
         # if the token is in between both entities
         if r2 < l:
            lemma = tree.get_lemma(t).lower()
            # No word increased the score here
            if lemma in []: 
               return 'mechanism'
            if lemma in [] : 
               return 'advise'
            if lemma in [] : 
               return 'int'
            if lemma in [] : 
               return 'effect'
   return None

## -- check pattern:  LCS of both entities is a verb with a should child,
# return ’advise’ if it is
def check_LCS_verb(tree,tkE1,tkE2):

   if tkE1 is not None and tkE2 is not None:
      lcs = tree.get_LCS(tkE1,tkE2)
      if tree.get_tag(lcs)[0:2] == "VB":
         children_of_lcs = tree.get_children(lcs)
         for child in children_of_lcs:
            lemma = tree.get_lemma(child).lower()
            if lemma in ['should']:
               return 'advise' 
   return None

# check if the parent of both entities is the same
def check_parent(tree,tkE1,tkE2,entities,e1,e2):
    if tkE1 is not None and tkE2 is not None:
      # get actual start/end of both entities
      l1,r1 = entities[e1]['start'],entities[e1]['end']
      l2,r2 = entities[e2]['start'],entities[e2]['end']
      p = []
      parent1 = tree.get_parent(tkE1)
      parent2 = tree.get_parent(tkE2)
      if (parent1==parent2):
          return 'advise'
    return None