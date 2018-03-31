import _pickle as pickle
import os
import pandas as pd


def main():
	train_path = "../../data/sogou_news_csv/train.csv"
	test_path = "../../data/sogou_news_csv/test.csv"

	train_pkl = "../../data/sogou_news_csv/train.pkl"
	test_pkl = "../../data/sogou_news_csv/test.pkl"

	if os.path.isfile(train_path) and os.path.isfile(test_path):
		print("Read: " + train_path)
		train_data = pd.read_csv(train_path, names=['class','title','content'])

		print("Read: " + test_path)
		test_data = pd.read_csv(test_path, names=['class','title','content'])

		train_data = clean(train_data)
		test_data = clean(test_data)


		print("Write: " + train_pkl)
		with open(train_pkl,'wb') as f:
			pickle.dump(train_data, f)

		print("Write: " + test_pkl)
		with open(test_pkl,'wb') as f:
			pickle.dump(test_data, f)

	else:
		print("Can't find train.csv or test.csv for sogou_news_csv")



def clean(data):
	data['title'].fillna(' ', inplace=True) # some missing title values - so replace with " "
	data['content'].fillna(' ', inplace=True)

	data['title'] = data['title'].replace("\"",'')
	data['content'] = data['content'].replace("\"", '')
	data['class'] = data['class'].replace("\"", '')

	data['title'] = data['title'] + " " + data['content']
	data.drop(['content'],axis=1,inplace=True)

	return data.as_matrix()


if __name__ == "__main__":
	main()