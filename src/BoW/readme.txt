loader.py:

For each of the datasets, define a function to read and return train_x, train_y, test_x, test_y

bag_of_words.py:

DBPedia doesn't use sparse matrices, so it is its own function (will change this later). Every other dataset uses
sparse_log_reg to do the training. CountVectorize the data and using LogisticRegression to classify