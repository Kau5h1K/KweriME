import pandas as pd
import os
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from textstat.textstat import textstat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report





SO_DATASET = os.path.join(os.path.dirname(__file__), 'data', 'stackoverflow_dataset.csv')
SO_ANSWERS = os.path.join(os.path.dirname(__file__), 'answer.csv')
TEST_DATA  = os.path.join(os.path.dirname(__file__), 'test.csv')
data = TEST_DATA
def load_so_corpus(dataset):
	return pd.read_csv(dataset)

def load_so_answers():
	return pd.read_csv(SO_ANSWERS, index_col=0)

def removehtmltags(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

def testdata():
    df = load_so_corpus(SO_DATASET)
    df[18:22].to_csv("test.csv")
    
    
def do_data_preprocessing(data):

    col = ['READABILITY','ANSWER_SCORE','ANSWERER_SCORE','SIMILARITY','TIME_DIFF','POLARITY','LABEL']
    final_feature_df = pd.DataFrame(columns = col)
    df = load_so_corpus(data)
    #print(df)
    print(df.describe()) 
    print(df.info())

    tokenize = lambda doc: doc.lower().split(" ")
    tfidf_vector = TfidfVectorizer(norm = 'l2',min_df = 0,stop_words = "english",use_idf = True,ngram_range=(1,3),sublinear_tf=True,tokenizer = tokenize)

    for answer in range(df.shape[0]):
        comment_text = df['COMMENT'][answer]
        
        if(type(comment_text) != float):
            blob = TextBlob(comment_text)
            polarity = blob.polarity
        else:
            polarity = 0

        label = (1 if df["A_ID"][answer] == df["Accepted_Answer_ID"][answer] else 0)
        
        answer_body = df["A_BODY"][answer]
        soup = BeautifulSoup(answer_body, "lxml")

        try:
            if soup.find('code'):
                for _ in range(len(soup.find_all('code'))):
                    if soup.find('code'):
                        soup.find('code').replaceWith('A good working code')
                    else:
                        break

            if soup.find('a'):
                for _ in range(len(soup.find_all('a'))):
                    if soup.find('a'):
                        soup.find('a').replaceWith('A working link')
                    else:
                        break
        except Exception as e:
			
            print ("answer: " + df["A_BODY"][answer])
            raise e

        if soup != "":
            readability = textstat.flesch_reading_ease(soup.text)
        else:
            if label:
                readability = 100.0
            else:
                readability = 0.0

        question_body = df['Q_BODY'][answer]
        answer_body = df['A_BODY'][answer]
        
        q_text = removehtmltags(question_body)
        a_text = removehtmltags(answer_body)
        
        QA_vector= tfidf_vector.fit_transform([q_text,a_text])
		
        Q_vector = QA_vector[0].toarray()
        A_vector = QA_vector[1].toarray()

        cs = cosine_similarity(Q_vector,A_vector)
		
        similarity = cs[0][0]

        if (df["U_REPUTATION"][answer]):
            final_feature_df.loc[answer]  = [readability,df["A_SCORE"][answer], df["U_REPUTATION"][answer],similarity,df['TIME_DIFF'][answer],polarity,label]
        else:
            final_feature_df.loc[answer]  = [readability,df["A_SCORE"][answer],0,similarity,df['TIME_DIFF'][answer],polarity,label]
    		

    final_feature_df.to_csv("answer1.csv")

def predict_label():
    dtm = load_so_answers()
    dtm.isnull().any()
    dtm = dtm.fillna(method='ffill')   
    accuracy_list = []	
    train_data = dtm[:(dtm.shape[0] - 1000)]	
    test_data = dtm[(dtm.shape[0] - 1000):]	
    X_train = train_data.drop('LABEL', 1)
   # X_train.drop('ANSWERER_SCORE',axis = 1,inplace = True)
    #X_train.drop('ANSWER_SCORE',axis = 1,inplace = True)	
    y_train = train_data['LABEL'].values	
    X_test = test_data.drop('LABEL', 1)
   # X_test.drop('ANSWERER_SCORE',axis = 1,inplace = True)
    #X_test.drop('ANSWER_SCORE',axis = 1,inplace = True)		
    y_test = test_data['LABEL'].values
    """--------------------------------------------------------------------------"""
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier()
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("Prediction accuracy of Dummy Classifier is %2.2f"%(accuracy_score(y_test,y_pred1)*100))
    from sklearn.metrics import f1_score
    print("F1 score of Dummy Classifier Testing is %2.2f" %f1_score(y_test,y_pred1,average= 'binary'))
    accuracy_list.append(accuracy_score(y_test,y_pred1)*100)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred1)
    print("The confusion matrix for Dummy Classifier testing is \n",cm)
    dum_roc_auc = roc_auc_score(y_test,y_pred1)
    print ("Dummy Classifier AUC = %2.2f" % dum_roc_auc)
    print(classification_report(y_test, y_pred1))
    print(pd.DataFrame(clf.predict_proba(X_test),columns=clf.classes_))
    #print(clf.score(X_test,y_test))

    """--------------------------------------------------------------------------"""
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred2 = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("Prediction accuracy of Naive Bayes is %2.2f"%(accuracy_score(y_test,y_pred2)*100))
    from sklearn.metrics import f1_score
    print("F1 score of Naive Bayes Testing is %2.2f" %f1_score(y_test,y_pred2,average= 'binary'))
    accuracy_list.append(accuracy_score(y_test,y_pred2)*100)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred2)
    print("The confusion matrix for Naive Bayes testing is \n",cm)
    nb_roc_auc = roc_auc_score(y_test,y_pred2)
    print ("Naive Bayes AUC= %2.2f" % nb_roc_auc)
    print(classification_report(y_test, y_pred2))
    print(pd.DataFrame(clf.predict_proba(X_test),columns=clf.classes_))
    """--------------------------------------------------------------------------"""
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth = 5,class_weight="balanced",
    min_weight_fraction_leaf=0.01)
    clf = clf.fit(X_test, y_test)
    importances = clf.feature_importances_
    feat_names = dtm.drop(['LABEL'],axis=1).columns
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12,6))
    plt.title("Feature importances by DecisionTreeClassifier")
    plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
    plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
    plt.xlim([-1, len(indices)])
    plt.show()    
    y_pred3 = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("Prediction accuracy of Decision Tree is %2.2f"%(accuracy_score(y_test,y_pred3)*100))
    from sklearn.metrics import f1_score
    print("F1 score of Decision Tree Testing is %2.2f" %f1_score(y_test,y_pred3,average= 'binary'))
    accuracy_list.append(accuracy_score(y_test,y_pred3)*100)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred3)
    print("The confusion matrix for Decision Tree testing is \n",cm)
    dt_roc_auc = roc_auc_score(y_test,y_pred3)
    print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
    print(classification_report(y_test, y_pred3))
    print(pd.DataFrame(clf.predict_proba(X_test),columns=clf.classes_))

    """--------------------------------------------------------------------------"""
    """from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=7)
    clf.fit(X_train, y_train)
    y_pred4 = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("Prediction accuracy of Random Forest Classifier is %2.2f"%(accuracy_score(y_test,y_pred4)*100))
    from sklearn.metrics import f1_score
    print("F1 score of Random Forest Classifier Testing is %2.2f" %f1_score(y_test,y_pred4,average= 'binary'))
    accuracy_list.append(accuracy_score(y_test,y_pred4)*100)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred4)
    print("The confusion matrix for Random Forest Classifier testing is \n",cm)
    rf_roc_auc = roc_auc_score(y_test,y_pred4)
    print ("Random forests AUC = %2.2f" % rf_roc_auc)
    print(classification_report(y_test, y_pred4))
    print(pd.DataFrame(clf.predict_proba(X_test),columns=clf.classes_))
    """
    """--------------------------------------------------------------------------"""
    fig,ax = plt.subplots()
    x = np.arange(3)
    plt.bar(x,accuracy_list,0.6)
    plt.xticks(x,('Baseline','Naive Bayes','Decision Tree'),fontsize = 12)
    ax.set_xlabel('Classifiers',fontsize = 14,weight= 'bold')
    ax.set_ylabel('Accuracy',fontsize = 14,weight = 'bold')
    plt.show()
    """--------------------------------------------------------------------------"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
    nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, y_pred2)
    dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test,y_pred3)
    #rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, y_pred4)
    
    plt.figure()
    
    plt.plot(fpr, tpr, label='Dummy Classifier (area = %0.2f)' % dum_roc_auc)
    
    plt.plot(nb_fpr, nb_tpr, label='Naive Bayes (area = %0.2f)' % nb_roc_auc)
    
    plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)
    
    #plt.plot(rf_fpr, rf_tpr, label='Random Forests (area = %0.2f)' % rf_roc_auc)
    
    # Plot Base Rate ROC
    plt.plot([0,1], [0,1],label='Base Rate' 'k--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Graph')
    plt.legend(loc="lower right")
    plt.show()
    testdata()
    
def nb():
    dtm = load_so_answers()
    dtm.isnull().any()
    dtm = dtm.fillna(method='ffill')   
    train_data = dtm[:(dtm.shape[0] - 1000)]	
    X_train = train_data.drop('LABEL', 1)
   # X_train.drop('ANSWERER_SCORE',axis = 1,inplace = True)
    #X_train.drop('ANSWER_SCORE',axis = 1,inplace = True)	
    y_train = train_data['LABEL'].values	
   # X_test.drop('ANSWERER_SCORE',axis = 1,inplace = True)
    #X_test.drop('ANSWER_SCORE',axis = 1,inplace = True)		
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf    
    
def dt():
    dtm = load_so_answers()
    dtm.isnull().any()
    dtm = dtm.fillna(method='ffill')   
    train_data = dtm[:(dtm.shape[0] - 1000)]
    test_data = dtm[(dtm.shape[0] - 1000):]		
    X_train = train_data.drop('LABEL', 1)
   # X_train.drop('ANSWERER_SCORE',axis = 1,inplace = True)
    #X_train.drop('ANSWER_SCORE',axis = 1,inplace = True)	
    y_train = train_data['LABEL'].values	
   # X_test.drop('ANSWERER_SCORE',axis = 1,inplace = True)
    #X_test.drop('ANSWER_SCORE',axis = 1,inplace = True)
    X_test = test_data.drop('LABEL', 1)
   # X_test.drop('ANSWERER_SCORE',axis = 1,inplace = True)
    #X_test.drop('ANSWER_SCORE',axis = 1,inplace = True)		
    y_test = test_data['LABEL'].values		
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth = 5,class_weight="balanced",
    min_weight_fraction_leaf=0.01)
    clf = clf.fit(X_test, y_test)
    return clf    
    
def predict_test():
    dtm = pd.read_csv(os.path.join(os.path.dirname(__file__), 'answer1.csv'), index_col=0)
    clf = nb()
    y_pred = clf.predict(dtm.values[:,:-1])
    prob = pd.DataFrame(clf.predict_proba(dtm.values[:,:-1]),columns=clf.classes_)
    print("The probabilities of each answer on non-accepted and accepted classes are :\n",prob)
    idx = prob.index[np.where(prob[1] == max(prob[1]))].tolist().pop()
    print("KweriME predicts the best answer to be answer number =",idx +1,"out of given",len(prob.index),"answers",)
    
def main():
    do_data_preprocessing(data)
    testdata()
    predict_label()
    predict_test()

if __name__ == '__main__':
    main()
