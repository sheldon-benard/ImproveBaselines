import _pickle as pickle
import os
import pandas as pd


def main():
	train_path = "../../data/yahoo_answers_csv/train.csv"
	test_path = "../../data/yahoo_answers_csv/test.csv"

	train_pkl = "../../data/yahoo_answers_csv/train.pkl"
	test_pkl = "../../data/yahoo_answers_csv/test.pkl"

	if os.path.isfile(train_path) and os.path.isfile(test_path):
		print("Read: " + train_path)
		train_data = pd.read_csv(train_path, names=['class','title','question_content','best_answer'])

		print("Read: " + test_path)
		test_data = pd.read_csv(test_path, names=['class','title','question_content','best_answer'])

		train_data = clean(train_data)
		test_data = clean(test_data)


		print("Write: " + train_pkl)
		with open(train_pkl,'wb') as f:
			pickle.dump(train_data, f)

		print("Write: " + test_pkl)
		with open(test_pkl,'wb') as f:
			pickle.dump(test_data, f)

	else:
		print("Can't find train.csv or test.csv for amazon_review_full")



def clean(data):
	# some missing title values - so replace with " "
	data['title'].fillna(' ', inplace=True)
	data['question_content'].fillna(' ', inplace=True)
	data['best_answer'].fillna(' ', inplace=True)

	data['title'] = data['title'].replace("\"",'')
	data['question_content'] = data['question_content'].replace("\"", '')
	data['class'] = data['class'].replace("\"", '')
	data['best_answer'] = data['best_answer'].replace("\"", '')

	data['title'] = data['title'] + " " + data['question_content'] + " " + data['best_answer']
	data.drop(['question_content'],axis=1,inplace=True)
	data.drop(['best_answer'], axis=1, inplace=True)

	return data.as_matrix()


if __name__ == "__main__":
	main()