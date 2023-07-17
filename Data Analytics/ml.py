from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn_evaluation import plot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # plotting
import pickle

import pandas as pd
def jointoken(token):
    join=[]
    for i in range (len(token)):
        string1=" "
        join.append(string1.join(token[i]))
    return join

def train_test(x,y):
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1234)
    return X_train,X_test,y_test,y_train

def bow_unigram(X_train,X_test):
    bow = CountVectorizer()
    X_train_uni = bow.fit_transform(X_train)
    X_test_uni = bow.transform(X_test)

    print('shape of X_train_bow is {}'.format(X_train_uni.get_shape()))
    print('shape of X_test_bow is {}'.format(X_test_uni.get_shape()))
    return X_train_uni,X_test_uni

def tfid_vector(X_train,X_test):
    #applying bow on x_train and x_test
    vectorizer = TfidfVectorizer()
    # we use the fitted CountVectorizer to convert the text to vector
    X_train_tfidf = vectorizer.fit_transform(X_train)
    #X_cv_tfidf = vectorizer.transform(X_cv)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf,X_test_tfidf

def bow_bigram(X_train,X_test):
    bi_gram = CountVectorizer(ngram_range=(1,2))
    X_train_bi = bi_gram.fit_transform(X_train)
    X_test_bi = bi_gram.transform(X_test)
    return X_train_bi,X_test_bi	

def optimalAlpha(X_train, y_train):
    NB = MultinomialNB()
    alpha_value = {'alpha':[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]} #params we need to try on classifier
    gsv = GridSearchCV(NB,alpha_value,cv=10,verbose=1,scoring='f1_micro')
    gsv.fit(X_train,y_train)
    print("Best HyperParameter: ",gsv.best_params_)
    print(gsv.best_score_)
    optimal_alpha=gsv.best_params_['alpha']
    plot.grid_search(gsv.cv_results_, change='alpha', kind='bar')
    plt.show()
    return optimal_alpha

def plot_roc(X_train, y_train, X_test,y_test,filename, figsize=(17, 6)):

    optimal_alpha = optimalAlpha(X_train,y_train)
    naive_opt = MultinomialNB(alpha=optimal_alpha)
    naive_opt.fit(X_train, y_train)#.predict_proba(X_test_bow)
    filename = filename
    pickle.dump(naive_opt, open(filename, 'wb'))
    y_pred = naive_opt.predict(X_test)
    # structures
    train_fpr, train_tpr, thresholds = roc_curve(y_train, naive_opt.predict_proba(X_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, naive_opt.predict_proba(X_test)[:,1])

    plt.grid(True)
    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.legend()
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.title("ROC CURVE FOR OPTIMAL K")
    plt.show()

    #Area under ROC curve
    print('Area under train roc {}'.format(auc(train_fpr, train_tpr)))
    print('Area under test roc {}'.format(auc(test_fpr, test_tpr)))

    print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))
    print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))
    print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(2),range(2))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g')

def train_logistic(X_train,y_train,X_test,y_test,C,solver,penalty,filename):
  #optimal_C = optimal_c(X_train, y_train,solver,penality)
  LR= LogisticRegression(C=C, solver=solver,penalty=penalty,max_iter=350)
  LR.fit(X_train,y_train)
  filename = filename
  pickle.dump(LR, open(filename, 'wb'))
  y_pred =LR.predict(X_test)
  print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))
  print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))
  print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))
  print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))
  df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(2),range(2))
  sns.set(font_scale=1.4)#for label size
  sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g')
  plt.show()
  # structures
  train_fpr, train_tpr, thresholds = roc_curve(y_train, LR.predict(X_train))
  test_fpr, test_tpr, thresholds = roc_curve(y_test, LR.predict(X_test))
  
  plt.grid(True)
  plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
  plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
  plt.legend()
  plt.xlabel("fpr")
  plt.ylabel("tpr")
  plt.title("ROC CURVE FOR OPTIMAL K")
  plt.show()

  #Area under ROC curve
  print('Area under train roc {}'.format(auc(train_fpr, train_tpr)))
  print('Area under test roc {}'.format(auc(test_fpr, test_tpr)))


def optimal_c(x_train,y_train):
  space = dict()
  space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
  space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
  space['C'] = [1e-5 , 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
  LR = LogisticRegression(max_iter=350)
  gsv = GridSearchCV(LR,space,cv=5,verbose=1,scoring='f1_weighted')
  gsv.fit(x_train,y_train)
  print("Best HyperParameter: ",gsv.best_params_)
  print(gsv.best_score_)
  optimal_C=gsv.best_params_
  plot.grid_search(gsv.cv_results_, change='C', kind='bar')
  plt.show()
  return optimal_C['C'],optimal_C['penalty'],optimal_C['solver']

def predict_data(X_test,y_test,model,data):
    df=pd.DataFrame()
    df['score']=y_test
    df['product_id']=data['productid']
    df['user_id']=data['userid']
    df['text']=data['text']
    df['token']=data['token']
    df['predict']=model.predict(X_test)

    return df
