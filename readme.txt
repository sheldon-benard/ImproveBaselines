Setup:

Setup virtualenv
    'virtualenv .'

Activate virtualenv
    'source bin/activate'

Install pre-req
    'pip install -r requirements.txt'

Install punkt for nltk:
    'python' - open python prompt
    'import nltk'
    'nltk.download('punkt')'


1. DBPedia
    Download: dbpedia_csv.tar.gz
    Unzip and put dbpedia_csv folder in the 'data' folder of the project

    From the project root:
        'cd src/Preprocess'
        'python pre_DBPedia.py'
        'cd ../BoW'
        'python bag_of_words.py DBPedia'


    Current accuracy: 0.975

2. Amazon full review
    Download: amazon_review_full_csv.tar.gz
    Unzip and put amazon_review_full_csv folder in the 'data' folder of the project

    From the project root:
        'cd src/Preprocess'
        'python pre_AmazonFull.py'
        'cd ../BoW'
        'python bag_of_words.py amazon_full'

    Current Accuracy: 0.521169

3. Amazon polarity
   Download: amazon_review_polarity_csv.tar.gz
   Unzip and put amazon_review_polarity_csv folder in the 'data' folder of the project

    From the project root:
        'cd src/Preprocess'
        'python pre_AmazonPolarity.py'
        'cd ../BoW'
        'python bag_of_words.py amazon_polarity'

    Current Accuracy: 0.8880775

4. Yahoo answers
   Download: yahoo_answers_csv.tar.gz
   Unzip and put yahoo_answers_csv folder in the 'data' folder of the project

    From the project root:
        'cd src/Preprocess'
        'python pre_Yahoo.py'
        'cd ../BoW'
        'python bag_of_words.py yahoo'

    Current Accuracy: 0.6804

5. Sogou
   Download: sogou_news_csv.tar.gz
   Unzip and put sogou_news_csv folder in the 'data' folder of the project

    From the project root:
        'cd src/Preprocess'
        'python pre_Sogou.py'
        'cd ../BoW'
        'python bag_of_words.py sogou'

    Current Accuracy: 0.92975

6. AG
   Download: ag_news_csv.tar.gz
   Unzip and put ag_news_csv folder in the 'data' folder of the project

    From the project root:
        'cd src/Preprocess'
        'python pre_AG.py'
        'cd ../BoW'
        'python bag_of_words.py ag'

    Current Accuracy: 0.904736