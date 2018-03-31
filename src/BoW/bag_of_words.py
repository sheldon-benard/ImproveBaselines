import sys
from loader import loadDBPedia
from loader import loadAmazonFull
import nltk
import numpy as np
from scipy.sparse import csr_matrix,vstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score



def DBPedia():

	train_x,train_y,test_x,test_y = loadDBPedia()
	print("train_x",train_x.shape,"test_x",test_x.shape)
	features = 5000
	stop = 'english'

	vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize,stop_words=stop,max_features=features)
	train_grams = vectorizer.fit_transform(train_x).toarray()

	test_vectorizer = CountVectorizer(tokenizer = nltk.word_tokenize, vocabulary = vectorizer.vocabulary_, stop_words = stop)
	test_grams = test_vectorizer.fit_transform(test_x).toarray()

	X = np.asarray(train_grams)
	Y = np.asarray(train_y)

	X_test = np.asarray(test_grams)
	Y_test = np.asarray(test_y)

	print("training Logistic Regression")
	classifier = LogisticRegression()
	classifier.fit(X, Y)

	predicted = classifier.predict(X_test)
	score = accuracy_score(Y_test,predicted)

	print(score)


def amazon_full():
	train_x,train_y,test_x,test_y = loadAmazonFull()
	print("train_x",train_x.shape,"test_x",test_x.shape)
	features = 5000
	stop = 'english'

	print("Fit_transform")
	vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, stop_words=stop, max_features=features)
	train_grams = csr_matrix(vectorizer.fit_transform(train_x))

	train_x = None
	Y = np.asarray(train_y)

	print("LogReg fit")
	classifier = LogisticRegression()
	classifier.fit(train_grams, Y)

	train_y = None
	train_grams = None
	Y = None

	print("Test")
	test_grams = csr_matrix(vectorizer.transform(test_x))

	Y_test = np.asarray(test_y)

	predicted = classifier.predict(test_grams)
	test_grams = None

	score = accuracy_score(Y_test,predicted)

	print(score)



if __name__ == "__main__":
	if len(sys.argv) == 2:
		if sys.argv[1] == "DBPedia":
			DBPedia()
		if sys.argv[1] == "amazon_full":
			amazon_full()

	else:
		print("Provide 1 command-line argument to specify dataset")