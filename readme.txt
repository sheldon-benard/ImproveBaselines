Setup:

Data: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

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

BoW:

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



VW:

Download the data and put the data folders in the 'data' folder as outlined in the BoW section above

Install Vowpal Wabbit:

    (DEPENDENCIES)
        brew install libtool
        brew install autoconf
        brew install automake
        brew install boost
        brew install boost-python

    Navigate to 'VW' folder
        cd VW

    Make VW repo and test that it works
        git clone git://github.com/JohnLangford/vowpal_wabbit.git
        cd vowpal_wabbit
        make
        make test

    If you encounter issues: https://github.com/JohnLangford/vowpal_wabbit


Methodology:
- Remove punctuation (r'[^\w\s\d]|_')
- Remove stop words (any word at most 2 characters)
- lowercase
- Hyperparameters: bits, loss_function, ngram, skips


    ./vw --oaa 14 -d ../../dbpedia_csv/train_vm.csv --loss_function hinge -b25 -f ../../dbpedia_csv/model.vw
    ./vw -t ../../dbpedia_csv/test_vm.csv -i ../../dbpedia_csv/model.vw -p ../../dbpedia_csv/pred.txt


    ./vw --oaa 5 -d ../../../data/sogou_news_csv/train_vw.csv --loss_function hinge -b25 --ngram 2 -f ../../../data/sogou_news_csv/model.vw
    ./vw -t ../../../data/sogou_news_csv/test_vw.csv -i ../../../data/sogou_news_csv/model.vw -p ../../../data/sogou_news_csv/pred.txt

    ./vw --oaa 10 -d ../../yahoo_answers_csv/train_vm.csv --loss_function hinge -b25  -f ../../yahoo_answers_csv/model.vw
    ./vw -t ../../yahoo_answers_csv/test_vm.csv -i ../../yahoo_answers_csv/model.vw -p ../../yahoo_answers_csv/pred.txt

    ./vw --oaa 5 -d ../../amazon_review_full_csv/train_vm.csv --loss_function hinge -b25  --ngram 2 --skips 2 -f ../../amazon_review_full_csv/model.vw
    ./vw -t ../../amazon_review_full_csv/test_vm.csv -i ../../amazon_review_full_csv/model.vw -p ../../amazon_review_full_csv/pred.txt

    ./vw --oaa 2 -d ../../amazon_review_polarity_csv/train_vm.csv --loss_function hinge -b25 --ngram 2 --skips 2  -f ../../amazon_review_polarity_csv/model.vw
    ./vw -t ../../amazon_review_polarity_csv/test_vm.csv -i ../../amazon_review_polarity_csv/model.vw -p ../../amazon_review_polarity_csv/pred.txt

