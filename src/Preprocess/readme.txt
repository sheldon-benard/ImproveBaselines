Description of the preprocessing files:

The purpose is to convert the csv to pkl file, for faster loading. In each file, I use pandas to read the csv and,
depending on the number of columns, label the columns using 'names='

Make sure the data is in the 'data' folder:
    ex. DBPedia:

    DBPedia
    Download: dbpedia_csv.tar.gz
    Unzip and put dbpedia_csv folder in the 'data' folder of the project

    From the project root:
        'cd src/Preprocess'
        'python pre_DBPedia.py'
