# ./vw --oaa 5 -d ../../../data/sogou_news_csv/train_vw.csv --loss_function hinge -b25 --ngram 2 -f ../../../data/sogou_news_csv/model.vw
# ./vw -t ../../../data/sogou_news_csv/test_vw.csv -i ../../../data/sogou_news_csv/model.vw -p ../../../data/sogou_news_csv/pred.txt

# vowpal_wabbit/vowpalwabbit/vw --oaa 5 -d ../data/amazon_review_full_csv/train_vw.csv -c --passes 15 --loss_function logistic -b25 --ngram 2 --learning_rate 0.2 --skips 1 --holdout_period 3 -f model.vw
# vowpal_wabbit/vowpalwabbit/vw -t ../data/amazon_review_full_csv/test_vw.csv -i model.vw -p pred.txt

# vowpal_wabbit/vowpalwabbit/vw --oaa 2 -d ../data/amazon_review_polarity_csv/train_vw.csv -c --passes 15 --holdout_period 3 --learning_rate 0.3 --loss_function logistic -b25 --ngram 2 --skips 1 -f model.vw
# vowpal_wabbit/vowpalwabbit/vw -t ../data/amazon_review_polarity_csv/test_vw.csv -i model.vw -p pred.txt

# vowpal_wabbit/vowpalwabbit/vw --oaa 14 -d ../data/dbpedia_csv/train_vw.csv -c --loss_function logistic -b25 --passes 15 --holdout_period 3 --learning_rate 0.5 --ngram 2 --skips 1 -f model.vw
# vowpal_wabbit/vowpalwabbit/vw -t ../data/dbpedia_csv/test_vw.csv -i model.vw -p pred.txt

# vowpal_wabbit/vowpalwabbit/vw --oaa 5 -d ../data/sogou_news_csv/train_vw.csv --loss_function logistic -b25 --ngram 2 --skips 1 -f model.vw
# vowpal_wabbit/vowpalwabbit/vw -t ../data/sogou_news_csv/test_vw.csv -i model.vw -p pred.txt


import itertools
import subprocess
import time

def hyper_parameter_tuning(classes, dataset):
	print(dataset)
	data_folder = "../data/" + dataset + "/"

	bits = ['b25']
	loss = ['logistic']
	ngrams = ['1','2','3']
	skips = [None, '1']
	learn = [0.001,0.01,0.1,0.3,0.5,0.7,0.9]

	for b,l,n,s,lr in itertools.product(bits,loss,ngrams,skips,learn):
		#train
		oaa = " --oaa " + str(classes)
		train = " -d " + data_folder + "train_vw.csv"
		loss_function = " --loss_function " + l
		bit = " -" + b
		ngram = " --ngram " + str(n)
		skip = ""
		if s is not None:
			skip = " --skips " + str(s)

		learning_rate = " --learning_rate " + str(lr)

		model = " -f model.vw"

		cache = " -c"
		passes = " --passes 15"
		holdout = " --holdout_period 3"
		run = "Run:{" + ",".join([str(n),str(s),str(lr)]) + "}"
		print(run)
		with open(dataset + ".txt","a+") as f:
			f.write(run + "\n")
			time.sleep(1)
			cmd_train = "vowpal_wabbit/vowpalwabbit/vw" + oaa + train + cache + passes + holdout + learning_rate + loss_function + bit + ngram + skip + model
			subprocess.call(cmd_train,shell=True,stderr=f)
			f.write("\n\n")


# hyper_parameter_tuning(14,"dbpedia_csv")
# hyper_parameter_tuning(2,"amazon_review_polarity_csv")
# hyper_parameter_tuning(5,"amazon_review_full_csv")
# hyper_parameter_tuning(5,"sogou_news_csv")
hyper_parameter_tuning(10,"yahoo_answers_csv")