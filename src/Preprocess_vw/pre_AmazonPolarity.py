import _pickle as pickle
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def main():
	train_path = "../../data/amazon_review_polarity_csv/train.csv"
	test_path = "../../data/amazon_review_polarity_csv/test.csv"

	train_vm = "../../data/amazon_review_polarity_csv/train_vw.csv"
	test_vm = "../../data/amazon_review_polarity_csv/test_vw.csv"


	if os.path.isfile(train_path) and os.path.isfile(test_path):
		print("Read: " + train_path)
		train_data = pd.read_csv(train_path, names=['class','title','review'])

		print("Read: " + test_path)
		test_data = pd.read_csv(test_path, names=['class','title','review'])

		train_data = clean(train_data)
		test_data = clean(test_data)


		print("Write: " + train_vm)
		with open(train_vm,'w') as f:
			np.savetxt(f,train_data,fmt="%s")


		print("Write: " + test_vm)
		with open(test_vm,'w') as f:
			np.savetxt(f,test_data,fmt="%s")

	else:
		print("Can't find train.csv or test.csv for amazon_review_full")



def clean(data):
	data['title'] = data['title'].str.replace(r'[^\w\s\d]|_', '')
	data['review'] = data['review'].str.replace(r'[^\w\s\d]|_', '')

	data['title'].fillna(' ', inplace=True) # some missing title values - so replace with " "
	data['title'] = data['title'] + " " + data['review']
	data.drop(['review'],axis=1,inplace=True)

	data['title'] = data['title'].str.replace(r'\b\w{1,2}\b','') # remove 1 and 2 character words
	data['title'] = data['title'].str.lower() # lowercase everything

	data['class'] = data['class'].astype(str) + " | " + data['title']
	data.drop(['title'], axis=1, inplace=True)

	data = shuffle(data)

	return data.as_matrix()


if __name__ == "__main__":
	main()