get_centroids.py

This file obtains the bag-of-centroids features and runs logistic regression for all corpora. Arguments are:
--resume: an option to use pre-trained centroids and cluster assignments. This saves a lot of time, as running kmeans is costly, but can only be done if the module has already been run once. 
--predict: an option to predict from pre-trained models for each corpus (can be done if the module has been run before)
--multi: this option toggles which type of logistic regression generalization to use. "ovr" corresponds to one-vs-rest classification, and "multi" corresponds to multinomial logistic regression. 


Requirements
------------
In order to run, pre-trained Google News word embeddings must be downloaded and put in the ~/data folder. These can be found here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

Note that before running, all files must be converted to .pkl files. This can be done by following the instructions in ~/src/Preprocess. 

Usage
-----
this file can be run from the ~/src/BoC directory via the following command:
`python get_centroids.py` 

Options can be viewed with 
`python get_centroids.py --help`
