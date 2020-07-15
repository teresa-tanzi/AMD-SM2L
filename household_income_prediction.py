import os
from pyspark import SparkContext
from pyspark.sql import SQLContext
from utils import load_preprocessed_dataframe, parse_point
from ridge_regression import RidgeRegression

if __name__ == '__main__':
    data_path = './data/2013-american-community-survey'

    sc = SparkContext('local[*]')
    sqlContext = SQLContext(sc)

    label = 'HINCP'
    df = load_preprocessed_dataframe(sqlContext, label)

    print('Casting data points into LabeledPoints...')
    parsed_data = df.rdd.map(lambda s: parse_point(s, label))
    print('Preprocessing... DONE!')

    # Divide data in training, validatioin and test set
    print('Subdividing data into train, validation and test set...')

    weights = [.8, .1, .1]
    seed = 42

    parsed_train_data, parsed_val_data, parsed_test_data = parsed_data.randomSplit(weights, seed = seed)
    parsed_train_data.cache()
    parsed_val_data.cache()
    parsed_test_data.cache()

    print('DONE!\nNow we can train our model!')

    # Train the model on the test set
    n_iters = 100
    reg_factor = 1e-10
    learn_rate = 0.01

    print('Training the model with parameters:\n\tNumber of iterations = {}\n\tLearning rate = {}\n\tRegularization factor = {}\n...'
        .format(n_iters, learn_rate, reg_factor))

    model = RidgeRegression(n_iters, learn_rate, reg_factor)
    model.fit(parsed_train_data)

    print('Training... DONE!')
    print('Training RMSE = {}'.format(model.train_error[-1]))

    # Evaluate the model on the validation set
    labels_and_preds_val = parsed_val_data.map(lambda p: model.predict_with_label(model.weights, p))
    rmse_val = model.rmse(labels_and_preds_val)

    print('Validation RMSE: = {}'.format(rmse_val))