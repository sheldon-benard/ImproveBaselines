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

1. Amazon full review
    Download: amazon_review_full_csv.tar.gz
    Unzip and put amazon_review_full_csv folder in the 'data' folder of the project

    From the project root:
        'cd src/Preprocess'
        'python pre_AmazonFull.py'
        'cd ../BoW'
        'python bag_of_words.py amazon_full'