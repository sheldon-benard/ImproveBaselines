import _pickle as pickle
import os
import pandas as pd


def main():
	train_path = "../../data/dbpedia_csv/train.csv"
	test_path = "../../data/dbpedia_csv/test.csv"

	train_pkl = "../../data/dbpedia_csv/train.pkl"
	test_pkl = "../../data/dbpedia_csv/test.pkl"

	if os.path.isfile(train_path) and os.path.isfile(test_path):
		print("Read: " + train_path)
		train_data = pd.read_csv(train_path, names=['class','title','abstract'])

		print("Read: " + test_path)
		test_data = pd.read_csv(test_path, names=['class','title','abstract'])

		train_data = clean(train_data)
		test_data = clean(test_data)


		print("Write: " + train_pkl)
		with open(train_pkl,'wb') as f:
			pickle.dump(train_data, f)

		print("Write: " + test_pkl)
		with open(test_pkl,'wb') as f:
			pickle.dump(test_data, f)

	else:
		print("Can't find train.csv or test.csv for DBPedia")



def clean(data):
	# For DBPedia: "Before feeding the data to the models, we concatenate the
	# title and short abstract together to form a single input for
	# each sample.
	data['title'] = data['title'].replace("\"",'')
	data['abstract'] = data['abstract'].replace("\"", '')
	data['title'] = data['title'] + " " + data['abstract']
	data.drop(['abstract'],axis=1,inplace=True)

	return data.as_matrix()


if __name__ == "__main__":
	main()