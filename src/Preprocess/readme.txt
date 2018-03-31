Description of the preprocessing files:

The purpose is to convert the csv to pkl file, for faster loading. In each file, I use pandas to read the csv and,
depending on the number of columns, label the columns using 'names='

The clean() attempts to remove " and fill in values that are missing, using fillna(). Then, concatenate the content columns, similar
to what the original authors did