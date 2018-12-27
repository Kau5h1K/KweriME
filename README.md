# KweriME 
- A Q&amp;A based model which predicts the accepted answers
of questions in CQA sites”


A python 2.7 compiler or above is needed to compile and run
the program. Suggested IDE is Spyder.

The stated accuracy and results are obtained using 1 lakh records.
The accuracy and results might differ when using smaller training dataset.

Additionaly, the following libraries should be installed:
nltk, sklearn, numpy, pandas, Textblob,Textstat,Beautifulsoup4

You will also need to install NLTK data if you
do not have it on your system(running nltk for the first time).
http://www.nltk.org/data.html
http://www.nltk.org/install.html

Model Classifier prediction and plotting results
- To generate the graphs a seperate file 'showviz.py' is used.
- In the code the test and actual results after feature engineering needs to be stored in the file
under the name test_data.csv and answer.csv respectively.
- The 'feature_eng.py' file will use the stackoverflow_dataset.csv and test_data.csv files and plot the accuracy and other performance evaluation metrics along with it
