from sklearn.metrics import accuracy_score

dataset = "dbpedia_csv"
# dataset = "amazon_review_polarity_csv"
# dataset = "amazon_review_full_csv"
# dataset = "ag_news_csv"
# dataset = "sogou_news_csv"
# dataset = "yahoo_answers_csv"

test = "../data/" + dataset + "/test_vw.csv"

pred = None
true = None
with open('pred.txt','r') as f:
	pred = [int(label) for label in f.readlines()]

with open(test,'r') as f:
	true = [int(label.split(" | ")[0].strip()) for label in f.readlines()]

print(accuracy_score(true,pred))