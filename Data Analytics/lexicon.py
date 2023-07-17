import os
import pandas as pd
from afinn import Afinn
from tqdm import tqdm

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

NRC_lexicon=os.path.join(THIS_FOLDER, 'lexicon/NRCEmotion.xlsx')

Yelp_lexicon=os.path.join(THIS_FOLDER,'lexicon/YelpLexicon.txt')

def affin(df):
    afinn = Afinn()
    df['afinn'] = df["text"].apply(afinn.score)
    # passo in -1, 1
    df_afinn=df
    df_afinn['score'] = df_afinn['score'].astype(int) #control
    df_afinn['afinn'] = df_afinn['afinn'].astype(int)

    df_afinn.loc[df_afinn['score'] <=3, 'score'] = 0
    df_afinn.loc[df_afinn['score'] >=4, 'score'] = 1

    df_afinn.loc[df_afinn['afinn'] <=0, 'afinn'] = 0
    df_afinn.loc[df_afinn['afinn'] >=1, 'afinn'] = 1

    return df_afinn

def yelp_lexicon():
    lexicon_df = pd.read_csv(Yelp_lexicon, sep="\t")
    lexicon_df.columns = ["term", "score", "Npos", "Nneg"]
    Neg_text = lexicon_df['term'].str.contains('[a-z]+_NEG$')
    for x in range(38381):
        if Neg_text[x] == True:
            lexicon_df.drop(x,inplace= True,axis=0 )

    # create a dict mapping word to value
    lexicon = {}
    # print(lexicon_df)
    # print(max(lexicon_df.score))	
    # print(min(lexicon_df.score))	

    #https://www.w3schools.com/python/ref_func_zip.asp

    for word, score, npos, nneg in zip(lexicon_df["term"], lexicon_df["score"], lexicon_df["Npos"],lexicon_df["Nneg"]):
      if score<-1.7:
        value = -1
      elif (score>-1.7 and score<1):
        value = 0 #i do not consider 0's 
      else:
        value=1
      lexicon[str(word).lower()] = value #lower case
    return lexicon

def scoreYelp(data):
  print("scoreYelp")
  token=data['token']
  lexicon=yelp_lexicon()
  print(token)
  score=[]
  n_pos=[]
  n_neg=[]
  n_neut=[]
  for sentence in tqdm(token):
    #print(sentence)
    neg=0
    pos=0
    neut=0
    sentiment = 0    
    leng=len(sentence)
    for x in range(leng): #split function: divides the string from spaces
        if sentence[x-1]=="not":
            word = sentence[x]+ "_NEGFIRST" #add NEG...
            sentiment += lexicon.get(word.lower()) if lexicon.get(word.lower()) is not None else 0 #+= operator
            #print(lexicon.get(word.lower()))
            if( lexicon.get(word.lower()) is not None and lexicon.get(word.lower())<0  ):
              neg+=1
            elif(lexicon.get(word.lower()) is not None and lexicon.get(word.lower())>0 ):
              pos+=1
            elif(lexicon.get(word.lower()) is not None and lexicon.get(word.lower())==0 ):
              neut+=1
            #print("word neg first",word ,":",sentiment)
        else:
            word = sentence[x]
            sentiment += lexicon.get(word.lower()) if lexicon.get(word.lower()) is not None else 0 #+= operator
            #print(lexicon.get(word.lower()))
            if(lexicon.get(word.lower()) is not None and lexicon.get(word.lower())<0):
              neg+=1
            elif(lexicon.get(word.lower()) is not None and lexicon.get(word.lower())>0 ):
              pos+=1
            elif(lexicon.get(word.lower()) is not None and lexicon.get(word.lower())==0 ):
              neut+=1
            #print("word: ",word,":", sentiment)
    score.append(sentiment)
    n_neg.append(neg)
    n_pos.append(pos)
    n_neut.append(neut)
    #print(sentiment)
  return score,n_pos,n_neg,n_neut

def predict_lexicon(data):
  df=pd.DataFrame()
  df['product_id']=data['productid']
  df['user_id']=data['userid']
  df['text']=data['text']
  df['token']=data['token']
  df['score']=data['score']
  df['predict'],df['n_pos'],df['n_neg'],df['n_neut']=scoreYelp(data)
  df.loc[df['predict'] >=1, 'predict'] = 1
  df.loc[df['predict'] <=0, 'predict'] = 0
  print(df)
  return df

def lexicon_nlc():
    lexicon_df = pd.read_excel(NRC_lexicon, engine="openpyxl")
    
    lexicon = {}

    #https://www.w3schools.com/python/ref_func_zip.asp

    for word, pos, neg in zip(lexicon_df["English Word"], lexicon_df["Positive"], lexicon_df["Negative"]):
        if pos:
            value = 1
        elif neg:
            value = 0 #i do not consider 0's 
        else:
            continue
        lexicon[str(word).lower()] = value #lower case
    return lexicon


def score_NLC(token):
  print("scoreNLC")
  df=pd.DataFrame()
  lexicon=lexicon_nlc()
  score=[]
  for sentence in tqdm(token):
    sentiment = 0
    for word in sentence: #split function: divides the string from spaces
        sentiment += lexicon.get(word.lower()) if lexicon.get(word.lower()) is not None else 0 #+= operator
    score.append(sentiment)
  return score