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

BoW: refer to src/BoW/readme.txt
BoC: refer to src/BoC/readme.txt

VW:

Install Vowpal Wabbit:

    (DEPENDENCIES):
        'brew install libtool'
        'brew install autoconf'
        'brew install automake'
        'brew install boost'
        'brew install boost-python'

    Navigate to 'VW' folder:
        'cd VW'

    Make VW repo and test that it works:
        'git clone git://github.com/JohnLangford/vowpal_wabbit.git'
        'cd vowpal_wabbit'
        'make'
        'make test'

    If you encounter issues, refer to: https://github.com/JohnLangford/vowpal_wabbit


For any dataset:
    1. Download the dataset and put the data folders in the 'data' folder as outlined in the BoW readme.txt
    2. Navigate to the VW preprocess and preprocess dataset:
        'cd src/Preprocess_vw'
        'python pre_{dataset}.py'
    3. Navigate to the VW folder; from root:
        'cd VW'

    Hyperparameter tuning:

    4. In tuning.py, uncomment the method call (at the bottom) of the dataset to run

    5. Run tuning.py (NOTE: VW output appended to the {dataset}.txt file in the VW folder; only do this is hyperparameter tuning):
        'python tuning.py'

    Train and test (still in the VW folder):
        In the command line, run the following according to your dataset:
        (Fill in oaa according to number of classes in dataset;lr,ng,sk according to tuning)

            'vowpal_wabbit/vowpalwabbit/vw --oaa {oaa} -d ../data/{dataset}/train_vw.csv -c --loss_function logistic -b25 --passes 15 --holdout_period 3 --learning_rate {lr} --ngram {ng} --skips {sk} -f model.vw'
            'vowpal_wabbit/vowpalwabbit/vw -t ../data/{dataset}/test_vw.csv -i model.vw -p pred.txt'

        To get accuracy, in get_accuracy.py, uncomment the dataset line that corresponds to desired dataset
        run:
            'python get_accuracy.py'

    Utilizing optimal passes (still in the VW folder):
        In the command line, run the following according to your dataset:
        (Fill in oaa according to number of classes in dataset;lr,ng,sk,passes according to tuning)

            'vowpal_wabbit/vowpalwabbit/vw --oaa {oaa} -d ../data/{dataset}/train_vw.csv -c --loss_function logistic -b25 --passes {passes} --holdout_off --learning_rate {lr} --ngram {ng} --skips {sk} -f model.vw'
            'vowpal_wabbit/vowpalwabbit/vw -t ../data/{dataset}/test_vw.csv -i model.vw -p pred.txt'

        To get accuracy, in get_accuracy.py, uncomment the dataset line that corresponds to desired dataset
        run:
            'python get_accuracy.py'


