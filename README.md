# Position-Classifier-for-NBA-Players-Based-on-Performance-Stats

NBA player statistics for the regular season are provided in this assignment (https://www.basketball-reference.com/leagues/NBA_2023_per_game.html).
NBA players must be categorized into five positions on the basketball court: shooting guard (SG), point guard (PG), small forward (SF), power forward (PF), and center (C). Based on the players' average performance per game during a regular season, classify them. You are given a CSV file called "nba_stats.csv" that contains the dataset. We recommend using pandas to load CSV files and process the data. Anything from the previously provided examples can be used.

## Tasks

1) Use one classification method (for example : Decision Tree/Naive Bayes/KNN/SVM/Neural-Networks) on the dataset. You can apply any of the methods explained in this instruction notebook or any other method in scikit-learn. You can also implement your own method. You can tune your model by using any combination of parameter values. Use 80% of the data for training and the rest for validation. Print out the training and validation set accuracy of the model. Also, print out the confusion matrix for both training and validation sets.

2)  Test the model on a test set(~100 sample) and you'll have access to this file after submission . For now, given an example test set as "dummy_test.csv", apply the model in 1 to the dummy test set and print the accuracy and confusion matrix on dummy test set.

3) Use the same model with the same parameters chosen in 1. However, instead of using 80%/20% train/test split, apply 10-fold stratified cross-validation. Print out the accuracy of each fold. Print out the average accuracy across all the folds. 
