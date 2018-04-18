import sys
from loader import loadDBPedia, loadAmazonFull, loadAmazonPolarity, loadYahoo,loadSogou, loadAG
import nltk
import numpy as np
from scipy.sparse import csr_matrix,vstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score



def sparse_log_reg(load,multi,max_iter):
	train_x,train_y,test_x,test_y = load()
	print("train_x",train_x.shape,"test_x",test_x.shape)
	features = 5000
	stop = 'english'

	print("Fit_transform")
	vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, stop_words=stop, max_features=features)
	train_grams = csr_matrix(vectorizer.fit_transform(train_x))

	train_x = None
	Y = np.asarray(train_y)

	print("LogReg fit")
	classifier = None
	if multi:
		classifier = LogisticRegression(multi_class='multinomial',solver='sag',max_iter=max_iter)
	else:
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
	if len(sys.argv) == 3:
		multi = False
		if sys.argv[2] == 'multi':
			multi = True
			print("Using multinomial Log Reg")


		if sys.argv[1] == "DBPedia":
			sparse_log_reg(loadDBPedia,multi,1500)
		if sys.argv[1] == "amazon_full":
			sparse_log_reg(loadAmazonFull,multi,1500)
		if sys.argv[1] == "amazon_polarity":
			sparse_log_reg(loadAmazonPolarity,multi,1500)
		if sys.argv[1] == "yahoo":
			sparse_log_reg(loadYahoo,multi,1500)
		if sys.argv[1] == "sogou":
			sparse_log_reg(loadSogou,multi,1500)
		if sys.argv[1] == "ag":
			sparse_log_reg(loadAG,multi,1500)


	else:
		print("Provide 2 command-line argument to specify dataset and multi/ovr")