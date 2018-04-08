import _pickle as pickle
import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np


def main():
	train_path = "../../data/yahoo_answers_csv/train.csv"
	test_path = "../../data/yahoo_answers_csv/test.csv"

	train_vm = "../../data/yahoo_answers_csv/train_vw.csv"
	test_vm = "../../data/yahoo_answers_csv/test_vw.csv"

	if os.path.isfile(train_path) and os.path.isfile(test_path):
		print("Read: " + train_path)
		train_data = pd.read_csv(train_path, names=['class','title','question_content','best_answer'])

		print("Read: " + test_path)
		test_data = pd.read_csv(test_path, names=['class','title','question_content','best_answer'])

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
	data['question_content'] = data['question_content'].str.replace(r'[^\w\s\d]|_', '')
	data['best_answer'] = data['best_answer'].str.replace(r'[^\w\s\d]|_', '')

	data['title'].fillna(' ', inplace=True)
	data['question_content'].fillna(' ', inplace=True)
	data['best_answer'].fillna(' ', inplace=True)

	data['title'] = data['title'] + " " + data['question_content'] + " " + data['best_answer']
	data.drop(['question_content'],axis=1,inplace=True)
	data.drop(['best_answer'], axis=1, inplace=True)

	data['title'] = data['title'].str.replace(r'\b\w{1,2}\b','') # remove 1 and 2 character words
	data['title'] = data['title'].str.lower() # lowercase everything

	data['class'] = data['class'].astype(str) + " | " + data['title']
	data.drop(['title'], axis=1, inplace=True)

	data = shuffle(data)

	return data.as_matrix()


if __name__ == "__main__":
	main()