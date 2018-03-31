from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans



if __name__ == '__main__':
	# read word2vec vectors
	w2v_mapping = KeyedVectors.load_word2vec_format("../../data/GoogleNews-vectors-negative300.bin", binary=True)
	# print(w2v_mapping.__dict__.keys())
	vectors = w2v_mapping.vectors
	n_clusters = len(w2v_mapping.vocab)/5 
	cluster_assignments = KMeans(n_clusters = n_clusters).fit_predict(vectors)