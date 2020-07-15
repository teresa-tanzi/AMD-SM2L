import progressbar
import numpy as np
from pyspark.mllib.linalg import DenseVector

class RidgeRegression:
    """
    Implementation of the ridge regression algorithm that finds the minimum of the objective function
        via stochastic gradient descent. Since it considers a regression problem, the objective function
        to be minimized is the square loss.
    """
    
    def __init__(self, n_iters = 10, learn_rate = 0.01, reg_factor = 1e-10):
        """
        Constructor of a RidgeRegression object. Initializes it with its hyperparameters.

        Args:
            n_iters (int): Number of iterations for the stochastic gradient descent procedure.
            learn_rate (float): Initial learning rate for the stochastic grandient descent procedure.
            reg_factor (float): Regularization factor for the ridge regression.
        """

        self.n_iters = n_iters
        self.learn_rate = learn_rate
        self.reg_factor = reg_factor

    def predict_with_label(self, weights, observation):
        """
        Computes the predicted label and associates it with the real label.
        
        Args:
            weigths (np array of double): Array containig the weights of the linear relatioship with
                the features of the data points.
            observation (LabeledPoint): Data point composed by its feautures and its label.
            
        Returns:
            tuple (float, float): Pair (true label, predicted label) relative to the given data point.
        """
        
        return (observation.label, weights.dot(DenseVector(observation.features)))

    def squared_error(self, label, prediction):
        """
        Computes the squared error for a prediction.
        
        Args:
            label (float): True label for this observation.
            prediction (float): Predicted label for this observation.
            
        Returns:
            float: Squared difference between the true label and the predicted label.
        """
        
        return (label - prediction) ** 2

    def rmse(self, labels_and_preds):
        """
        Computes the root mean squared error for an RDD of (true label, predicted label) tuples.
        
        Args:
            labels_and_preds (RDD of (float, float)): RDD of (true label, predicted label) tuples.
            
        Returns:
            float: Square root of the mean of the squared errors of all the tuples in the RDD.
        """
        
        return np.sqrt(labels_and_preds.map(lambda p: self.squared_error(*p)).mean())

    def gradient_summand(self, weights, lp):
        """
        Computes a summand in the gradient formulation of the square loss.
        
        Args:
            weigths (DenseVector): Vector containing the weights learnt by the model.
            lp (LabeledPoint): A single observation composed by its features and its label.
            
        Returns:
            DenseVector: Dot product between the weights vector and the feature vector, minus the
                label and multiplied again for the the feature vector.
        """
        
        return (weights.dot(DenseVector(lp.features)) - lp.label) * lp.features

    def fit(self, train_data):
        """
        Trains the model in order to find the best weight vector that minimizes the square loss.
            It does so with stochastic gradient descent procedure.
            
        Args:
            train_data (RDD of LabeledPoints): training set used in order to real the weight vector.
            
        Returns:
            RidgeRegression: object corresponding to the lerant predictor.
        """
        
        n = train_data.count()
        d = len(train_data.take(1)[0].features)
        w = np.zeros(d)

        train_error = np.zeros(self.n_iters)

        bar = progressbar.ProgressBar(maxval = self.n_iters,
                                      widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for i in range(n_iters):
            labels_and_preds_train = train_data.map(lambda p: self.predict_with_label(w, p))
            train_error[i] = self.rmse(labels_and_preds_train)

            gradient_sum = train_data.map(lambda lp: DenseVector(self.gradient_summand(w, lp))) \
                            .reduce(lambda x, y: x + y)
            gradient = gradient_sum + (self.reg_factor * w)

            learn_rate_i = self.learn_rate / (n * np.sqrt(i + 1))
            w -= learn_rate_i * gradient
            
            bar.update(i + 1)
            
        bar.finish()

        self.weights = w
        self.train_error = train_error

        return self

    def predict(self, features):
        """
        Computes the predicted label for a given observation.
        
        Args:
            features (vector of floats): Feature vector of a data point.
            
        Returns:
            float: predicted label relative to the given data point.
        """
        
        return self.weights.dot(DenseVector(features))
