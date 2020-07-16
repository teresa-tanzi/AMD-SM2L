import os
import numpy as np
import zipfile
from pyspark.sql.functions import udf, isnan, when, count, col
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, Imputer
from pyspark.mllib.regression import LabeledPoint
from ridge_regression import RidgeRegression

def install_kaggle():
    """
    Installs and setups Kaggle API, that are needed in order to download the dataset.
    """
    
    #os.system('pip install --user kaggle')
    os.system('pip3 install --user kaggle')

    os.environ["PATH"] += os.pathsep + "/home/teresa/.local/bin"
    #os.system('export PATH=$PATH:/home/teresa/.local/bin')

    #os.mkdir('~/.kaggle')    
    #os.mknod('~/.kaggle/kaggle.json')
    #f = open('~/.kaggle/kaggle.json', 'w')
    #f.write('{"username":"teresatanzi","key":"a64ec0d865925975d3318adf576216b7"}')
    #f.close()
    #os.chmod('~/.kaggle/kaggle.json', 600)

    if not os.path.exists('~/.kaggle'):
        os.system("""echo '{"username":"teresatanzi","key":"a64ec0d865925975d3318adf576216b7"}' >> ~/.kaggle/kaggle.json""")
        os.system('chmod 600 ~/.kaggle/kaggle.json')

    os.system('export KAGGLE_USERNAME=teresatanzi')
    os.system('export KAGGLE_KEY=a64ec0d865925975d3318adf576216b7')
    
    #os.environ["KAGGLE_USERNAME"] = "teresatanzi"
    #os.environ["KAGGLE_KEY"] = "a64ec0d865925975d3318adf576216b7"

def download_dataset(path):
    """
    Checks if the dataset has already been downloaded and, if not, downloads and unzips it.
    
    Args:
        path (string): path where the dataset will be provided.
    """
    
    if os.path.exists(path + '/ss13husa.csv') and os.path.exists(path + '/ss13husb.csv') and \
            os.path.exists(path + '/ss13pusa.csv') and os.path.exists(path + '/ss13pusb.csv'):
        print('Data already downloaded.')
        return

    # setup Kaggle API
    print('Installing Kaggle API...')
    install_kaggle()
    
    # dataset download
    print('Downloading the dataset...')
    os.system('mkdir ./data')
    os.system('kaggle datasets download census/2013-american-community-survey -p ./data')

    print('Unzipping the dataset...')
    with zipfile.ZipFile("./data/2013-american-community-survey.zip","r") as zip_ref:
        zip_ref.extractall(path)

    print('DONE! Data have been extracted in dir {}.'.format(path))

def scale_features(df):
    """
    Scales all the features of a DataFrame so that all the values belongs to a range of [0, 1].
    
    Note:
        If the DataFrame is big, this operation can take a lot of time to be completed.

    Args:
        df (PySpark DataFrame): DataFrame composed by all double features.

    Returns:
        PySpark DataFrame: DataFrame with all the feature normalized in the interval [0, 1]
    """
        
    columns_to_scale = df.columns

    assemblers = [VectorAssembler(inputCols = [col], outputCol = col + "_vec") for col in columns_to_scale]
    scalers = [MinMaxScaler(inputCol = col + "_vec", outputCol = col + "_scaled") for col in columns_to_scale]
    pipeline = Pipeline(stages = assemblers + scalers)

    df = pipeline.fit(df).transform(df)
    
    unlist = udf(lambda x: float(list(x)[0]), DoubleType())
    names = {x + "_scaled": x for x in columns_to_scale}

    df = df.select([unlist(col(c)).alias(names[c]) for c in names.keys()])
    
    return df

def preprocess (df, na_threshold, label, imputer_strategy = 'mean', feature_scaling = True):
    """
    Preprocess a PySpark DataFrame, dropping the categorical feature and the discarded labels,
        casting the remaining features in double, dealing with null values with imputation and 
        dropping the features with too many missing values, dropping also all the data points
        with missing label. Optionally, scales all the features of the DataFrame.
        
    Note:
        Feature scaling can slow up the process if the DataFrame is huge.

    Args:
        df (PySpark DataFrame): DataFrame read by the csv file.
        na_threshold (float between 0 and 1): threshold that establish if a feature should be dropped or not
            based on its percentage of null values.
        label (string): feature that corresponds to the chosen label to be predicted.
        imputer_strategy (string): it can be `mean` or `median`, based on the imputation strategy we choose.
        feature_scaling (boolean): if True, all the feature in the DataFrame are scaled to have values belonging
            in intervall [0, 1], but it slows the process.

    Returns:
        PySpark DataFrame: Restult of the processing of the original DataFrame.
    """    
    
    print("Dropping features...")
    df = df.drop('RT')
    
    #possible_labels = ['PERNP', 'PINCP', 'WAGP'] if inputPathA == 'ss13pusa.csv' else ['HINCP', 'FINCP']     
    possible_labels = ['HINCP', 'FINCP']    
    possible_labels.remove(label)
    
    for i in possible_labels:
        df = df.drop(i)

    print("Casting features to double...")    
    df = df.select([col(c).cast("double") for c in df.columns])
    
    # we should save max and min label in order to convert back 
    row1_max = df.agg({label: "max"}).collect()[0]
    row1_min = df.agg({label: "min"}).collect()[0]
    print("\tMax {}: {}\n\tMin {}: {}".format(label,
                                              row1_max["max(" + label + ")"],
                                              label,
                                              row1_min["min(" + label + ")"]))
        
    print("Dropping features with more than {}% null values...".format(int(na_threshold * 100)))
    n = df.count()
    null_df = df.select([(count(when(col(c).isNull(), c))/n).alias(c) for c in df.columns])
    
    scheme = df.columns
    null_distr = null_df.take(1)[0].asDict().values()
    
    for i in np.where(np.array(list(null_distr)) > na_threshold)[0]:
        df = df.drop(scheme[i])
        
    print('We reduced the number of features to {}.'.format(len(df.columns)))
    print('Dropping data points with null label...')
    df = df.filter(df[label].isNotNull())    

    print('Imputing missing values...')    
    imputer = Imputer()
    imputer.setInputCols(df.columns)
    imputer.setOutputCols(df.columns)
    imputer.setStrategy(imputer_strategy)

    df = imputer.fit(df).transform(df)
    
    if feature_scaling:
        print("Feature scaling...")
        df = scale_features(df)

    print("DONE!")
        
    return df

def save_preprocessed_dataframe(df):
    """
    Saves the DataFrame on disk as .csv file.
    
    Note:
        This operation can be really slow. It should be done once and only in case of multiple executions
            of the learning code, in order to avoid to recompute the preprocessing every time.
    
    Args:
        df (PySpark DataFrame): DataFrame that has to be saved on disk.
    """
    
    print("Saving preprocessed data...")
    base_dir = os.path.join('./data')
    input_path = os.path.join('preprocessed_data')
    file_name = os.path.join(base_dir, input_path)

    df.write.csv(file_name, header = True)
    
    print("Saving preprocessed data... DONE!")

def load_preprocessed_dataframe(sqlContext, label, data_path = './data/2013-american-community-survey'):
    """
    Loads preprocessed data from disk. If there are no already preprocessed data on disk, donwloads
        the dataset and preprocesses it.
        
    Args:
        sqlContext (PySpark SQLContext): PySpark SQL entry point, needed in order to create PySpark DataFrames.
        label (string): Name of the feature chosen as the label to be predicted.
        data_path (string): Path where the dataset will be downloaded.
        
    Returns:
        PySpark DataFrame: DataFrame containing the preprocessed dataset.
    """
    
    # check if we already have presaved preprocessed data
    if os.path.exists('./data/preprocessed_data'):
        print('Reading the preprocessed saved dataset...')
        base_dir = os.path.join('./data')
        input_path = os.path.join('preprocessed_data')
        file_name = os.path.join(base_dir, input_path)

        df = sqlContext.read.csv(file_name, header = True)
        n = df.count()
        header_list = df.columns

        print("\tNumber of columns: {}\n\tNumber of rows: {}".format(len(header_list), n))

        return df

    else:
        #checks if dataset is already been downloaded
        download_dataset(data_path) 

        # reading the data as DataFrames
        print('Reading the dataset...')        
        base_dir = os.path.join(data_path)
        input_path_a = os.path.join('ss13husa.csv')
        input_path_b = os.path.join('ss13husb.csv')
        file_name_a = os.path.join(base_dir, input_path_a)
        file_name_b = os.path.join(base_dir, input_path_b)

        df_a = sqlContext.read.csv(file_name_a, header = True)
        df_b = sqlContext.read.csv(file_name_b, header = True)

        df = df_a.union(df_b)
        n = df.count()
        header_list = df.columns

        print("Number of columns: {}\nNumber of rows: {}".format(len(header_list), n))

        # preprocessing
        na_threshold = .6

        print('Preprocessing with\n\tthreshold = {},\n\tlabel = {}\n...'.format(na_threshold, label))
        df = preprocess(df, na_threshold, label)

        # Change the following variable in True to save the preprocessed DataFrame
        # This has to be done in order to avoid re-preprocessing data in developing phase
        # BE CAREFUL! Saving data can take a lot of time
        save_preprocessed_data = True
        if save_preprocessed_data == True: save_preprocessed_dataframe(df)
            
        return df

def parse_point(row, label, intercept = True):
    """
    Converts a row of a pyspark dataframe into a LabeledPoint.
    
    Args:
        row (PySpark DataFrame row): Row of a DataFrame composed all of double values.
        label (string): name of the feature corresponding to the label to be predicted.
        intercept (boolean): If True, a feature with a constant value 1 is added in order to learn also
            the value of the intercept.

    Returns:
        LabeledPoint: The line is converted into a LabeledPoint, which consists of a label and
            features.
    """
    
    row_dict = row.asDict()
    label_value = row_dict[label]
    
    del row_dict[label]
    feature_list = list(row_dict.values())
    if intercept: feature_list.insert(0, 1)
    
    return LabeledPoint(label_value, feature_list)

def real_label(scaled_lbl):
    """
    Converts a normalized label into a label in USD.

    Args:
        scaled_lbl (float): Normalized value for the feature choosen as label (HINCP).

    Returns:
        float: Value for the choosen label (HINCP) measured in USD. It is obtained by reversing the
            normalization formula.
    """

    # Values found during the preprocessing of the dataset.
    max_lbl = 2090000.0
    min_lbl = -19770.0
    
    return (max_lbl - min_lbl) * scaled_lbl + min_lbl

def grid_search(train_set, val_set, n_iters_params, reg_factor_params, learn_rate_params):
    """
    Performs grid search in order to find the best hyperparameters.

    Args:
        train_set (RDD of LabeledPoints): Training set used to be used to train the model.
        val_set (RDD of LabeledPoints): Validation set used to be used to evaluate the model.
        n_iters_params (list of int): list of parameters for the number of iterations of the stochastic
            gradient descent procedure.
        reg_factor_params (list of float): list of parameters for the regularization factor of ridge
            regression.
        learn_rate_params (list of float): list of parameters for the learning rate of the stochastic
            gradient descent procedure.

    Returns:
        RidgeRegression: ridge regression model with the lowest rmse value on the validation set.
    """
    print('Performing grid search...')

    best_model = None
    best_rmse_val = float('inf')

    for n_iters in n_iters_params:
        for reg_factor in reg_factor_params:
            for learn_rate in learn_rate_params:
                print('Training the model with parameters:\n\tNumber of iterations: {}\n\tRegularization factor: {} \
                    \n\tLearning rate: {}'.format(n_iters, reg_factor, learn_rate))
                model = RidgeRegression(n_iters, learn_rate, reg_factor)
                model.fit(parsed_train_data)

                # Evaluate the model on the validation set
                labels_and_preds_val = parsed_val_data.map(lambda p: model.predict_with_label(model.weights, p))
                rmse_val = model.rmse(labels_and_preds_val)

                print('Validation RMSE: {}'.format(rmse_val))

                if rmse_val < best_rmse_val:
                    best_model = model
                    best_rmse_val = rmse_val

    return best_model
