import _pickle as pickle
import os
import numpy as np



def loadDBPedia():
	train_pkl = "../../data/dbpedia_csv/train.pkl"
	test_pkl = "../../data/dbpedia_csv/test.pkl"

	if os.path.isfile(train_pkl) and os.path.isfile(test_pkl):

		print("Read DBPedia pkl files")

		train_in = open(train_pkl, 'rb')
		test_in = open(test_pkl, 'rb')

		train = pickle.load(train_in)
		test = pickle.load(test_in)

		train_in.close()
		test_in.close()

		train,test = np.asarray(train,dtype=object),np.asarray(test,dtype=object)

		train_x = np.asarray(train[:,1],dtype=object)
		train_y = np.asarray(train[:,0],dtype=np.int16)

		test_x = np.asarray(test[:,1],dtype=object)
		test_y = np.asarray(test[:,0],dtype=np.int16)

		return train_x,train_y,test_x,test_y


	else:
		print("Can't find train.pkl or test.pkl for DBPedia")


def loadAmazonFull():
	train_pkl = "../../data/amazon_review_full_csv/train.pkl"
	test_pkl = "../../data/amazon_review_full_csv/test.pkl"

	if os.path.isfile(train_pkl) and os.path.isfile(test_pkl):

		print("Read Amazon_full pkl files")

		train_in = open(train_pkl, 'rb')
		test_in = open(test_pkl, 'rb')

		train = pickle.load(train_in)
		test = pickle.load(test_in)

		train_in.close()
		test_in.close()

		train,test = np.asarray(train,dtype=object),np.asarray(test,dtype=object)

		train_x = np.asarray(train[:,1],dtype=object)
		train_y = np.asarray(train[:,0],dtype=np.int16)

		test_x = np.asarray(test[:,1],dtype=object)
		test_y = np.asarray(test[:,0],dtype=np.int16)

		return train_x,train_y,test_x,test_y


	else:
		print("Can't find train.pkl or test.pkl for Amazon_full")


def loadAmazonPolarity():
	train_pkl = "../../data/amazon_review_polarity_csv/train.pkl"
	test_pkl = "../../data/amazon_review_polarity_csv/test.pkl"

	if os.path.isfile(train_pkl) and os.path.isfile(test_pkl):

		print("Read Amazon_Polarity pkl files")

		train_in = open(train_pkl, 'rb')
		test_in = open(test_pkl, 'rb')

		train = pickle.load(train_in)
		test = pickle.load(test_in)

		train_in.close()
		test_in.close()

		train,test = np.asarray(train,dtype=object),np.asarray(test,dtype=object)

		train_x = np.asarray(train[:,1],dtype=object)
		train_y = np.asarray(train[:,0],dtype=np.int16)

		test_x = np.asarray(test[:,1],dtype=object)
		test_y = np.asarray(test[:,0],dtype=np.int16)

		return train_x,train_y,test_x,test_y


	else:
		print("Can't find train.pkl or test.pkl for Amazon_Polarity")


def loadYahoo():
	train_pkl = "../../data/yahoo_answers_csv/train.pkl"
	test_pkl = "../../data/yahoo_answers_csv/test.pkl"

	if os.path.isfile(train_pkl) and os.path.isfile(test_pkl):

		print("Read Yahoo_Answers pkl files")

		train_in = open(train_pkl, 'rb')
		test_in = open(test_pkl, 'rb')

		train = pickle.load(train_in)
		test = pickle.load(test_in)

		train_in.close()
		test_in.close()

		train,test = np.asarray(train,dtype=object),np.asarray(test,dtype=object)

		train_x = np.asarray(train[:,1],dtype=object)
		train_y = np.asarray(train[:,0],dtype=np.int16)

		test_x = np.asarray(test[:,1],dtype=object)
		test_y = np.asarray(test[:,0],dtype=np.int16)

		return train_x,train_y,test_x,test_y


	else:
		print("Can't find train.pkl or test.pkl for Yahoo_Answers")


def loadSogou():
	train_pkl = "../../data/sogou_news_csv/train.pkl"
	test_pkl = "../../data/sogou_news_csv/test.pkl"

	if os.path.isfile(train_pkl) and os.path.isfile(test_pkl):

		print("Read sogou_news_csv pkl files")

		train_in = open(train_pkl, 'rb')
		test_in = open(test_pkl, 'rb')

		train = pickle.load(train_in)
		test = pickle.load(test_in)

		train_in.close()
		test_in.close()

		train,test = np.asarray(train,dtype=object),np.asarray(test,dtype=object)

		train_x = np.asarray(train[:,1],dtype=object)
		train_y = np.asarray(train[:,0],dtype=np.int16)

		test_x = np.asarray(test[:,1],dtype=object)
		test_y = np.asarray(test[:,0],dtype=np.int16)

		return train_x,train_y,test_x,test_y

	else:
		print("Can't find train.pkl or test.pkl for sogou_news_csv")

def loadAG():
	train_pkl = "../../data/ag_news_csv/train.pkl"
	test_pkl = "../../data/ag_news_csv/test.pkl"

	if os.path.isfile(train_pkl) and os.path.isfile(test_pkl):

		print("Read ag_news_csv pkl files")

		train_in = open(train_pkl, 'rb')
		test_in = open(test_pkl, 'rb')

		train = pickle.load(train_in)
		test = pickle.load(test_in)

		train_in.close()
		test_in.close()

		train,test = np.asarray(train,dtype=object),np.asarray(test,dtype=object)

		train_x = np.asarray(train[:,1],dtype=object)
		train_y = np.asarray(train[:,0],dtype=np.int16)

		test_x = np.asarray(test[:,1],dtype=object)
		test_y = np.asarray(test[:,0],dtype=np.int16)

		return train_x,train_y,test_x,test_y

	else:
		print("Can't find train.pkl or test.pkl for ag_news_csv")