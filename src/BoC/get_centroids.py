from loader import  loadDBPedia, loadAmazonFull, loadAmazonPolarity, loadYahoo,loadSogou, loadAG
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from nltk import word_tokenize
from scipy.sparse import lil_matrix, csr_matrix,vstack
import re
import os
import sys
import numpy as np
import pandas as pd
import time
import _pickle as pickle
import argparse

def read_data(path):
	data = pd.read_csv(path, header=None)
	return data

def get_boc(sents, n_clusters, word_to_cluster, vocab):
	#boc_sents = np.zeros((len(sents), n_clusters))
	boc_sents = lil_matrix((len(sents), n_clusters), dtype=np.float32)
	for i, s in enumerate(sents):
		splitsent = word_tokenize(s)
		for word in splitsent:
			if word in vocab:
				clust = word_to_cluster[word]
				try:
					boc_sents[i,clust] += 1
				except IndexError:
					boc_sents[i,clust] = 1
	return boc_sents.tocsr()

def get_all(resume, predict, multi):

	t0 = time.clock()
	print("Read: {}".format("../../data/GoogleNews-vectors-negative300.bin"))
	# read word2vec vectors
	w2v_mapping = KeyedVectors.load_word2vec_format("../../data/GoogleNews-vectors-negative300.bin", binary=True)
	loaders = [loadDBPedia, loadAmazonFull, loadAmazonPolarity, loadYahoo,loadSogou, loadAG]
	#for c_loader in [loadDBPedia, loadAmazonFull, loadAmazonPolarity, loadYahoo,loadSogou, loadAG]:
	names = [re.sub("load","", x.__name__) for x in loaders]

	loaders = [loadAG, loadDBPedia, loadYahoo, loadAmazonFull, loadAmazonPolarity]
	names = ["AG", "DBPedia", "Yahoo", "AmazonFull", "AmazonPolarity"]

	for i, c_loader in enumerate(loaders):		
		corpus_name = names[i]
		boc_path = os.path.join("..","..","data","{}_boc.pkl".format(corpus_name))
		model_path = os.path.join("models","{}_model.pkl".format(corpus_name))

		print("Read: data")
		train_x, train_y, test_x, test_y = c_loader()	
		
		vocab = w2v_mapping.vocab
		vectors = w2v_mapping.vectors
		idx_to_word = w2v_mapping.index2word
		n_clusters = 5000
	#	n_clusters = int(len(w2v_mapping.vocab)/5)
		t0 = time.clock()
		if not resume:
			print("Begin kmeans...")
			km = MiniBatchKMeans(n_clusters = n_clusters, init_size=2*n_clusters)
			cluster_assignments = km.fit_predict(vectors)

		else:
			print("Resume kmeans...")
			with open("kmeans.pkl", "rb") as f1:
				km = pickle.load(f1)
			with open("clusters.pkl", "rb") as f1:
				cluster_assignments = pickle.load(f1)
		
		centroids = km.cluster_centers_

		print("Write: kmeans.pkl")
		with open("kmeans.pkl", "wb") as f1:
			pickle.dump(km, f1)
		print("Write: clusters.pkl")
		with open("clusters.pkl", "wb") as f1:
			pickle.dump(cluster_assignments, f1)

		word_to_cluster = dict(zip(idx_to_word, cluster_assignments))

		t0 = time.clock()
		if os.path.exists("../../data/{}_boc.pkl".format(corpus_name)):
			print("Read: ../../data/{}_boc.pkl".format(corpus_name))
			with open("../../data/{}_boc.pkl".format(corpus_name), "rb") as f1:
				train_boc = pickle.load(f1)
		else:
			print("Begin train BoC...")
			train_boc = get_boc(train_x, n_clusters, word_to_cluster, vocab)
		print("Begin test BoC...")
		test_boc = get_boc(test_x, n_clusters, word_to_cluster, vocab)
		
		print("train_boc shape: {}".format(train_boc.shape))

		print("Write: {}".format(boc_path))
		with open(boc_path, "wb") as f1:
			pickle.dump(train_boc, f1)

		print("Finished BoCs in: {}".format(time.clock() - t0))
		print("Construct logistic regression...")
		t0 = time.clock()

		if not predict:

			clf = LogisticRegression(solver="saga", max_iter=100, multi_class=multi, verbose=1) 
			print("Begin fitting...")
			clf.fit(train_boc, train_y)

			print("Write: {}".format(model_path))
			with open(model_path, "wb") as f1:
				pickle.dump(clf, f1)	
		else:
			print("Read: {}".format(model_path))
			with open(model_path, "rb") as f1:
				clf = pickle.load(f1)

		print("Begin predict...")
		train_pred = clf.predict(train_boc)

		train_acc = accuracy_score(train_y, train_pred)
		test_pred = clf.predict(test_boc)

		acc= accuracy_score(test_y, test_pred)
		with open("results_ovr.txt", "a") as f1:
			f1.write("{},{},{}\n".format(corpus_name,train_acc, acc))

		print("Accuracy for {}: ".format(corpus_name), acc)



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="run bag-of-centroids logistic regression on all corpora")
	parser.add_argument("--resume", action="store_true", help="set to True if resuming from previously computed centroids/cluster assignments")
	parser.add_argument("--predict", action="store_true", help="Set to True if predicting (not training) from pre-trained model")
	parser.add_argument("--multi",  default="multi", help="set to ovr if one-vs-rest classification desired, default is multinomial logistic regression")
	args = parser.parse_args()

	multi = "multinomial" if args.multi=='multi' else "ovr"

	lst_args = [args.resume, args.predict, multi]
	get_all(*lst_args)
	








