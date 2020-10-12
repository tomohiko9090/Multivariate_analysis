"""
  Created on Saturday September 26 2020 at 10:10
  Author Keisuke Noji @CVSLab.

  * Lasso return
  * Use boston data
  * USe sklearn

"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse


def load_data():
    """
    Read BOSTON Data
    :return: explain_variable(12 columns)
             purpose variable(1D)
    """
    boston = datasets.load_boston()
    explain_variable = boston.data
    explain_label = ['CRIM', 'ZN', 'INDUS', 'CHARS', 'NOX', 'RM',
                     'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    explain_variable = pd.DataFrame(explain_variable,
                                    columns=explain_label)
    purpose_variable = boston.target
    purpose_variable = pd.DataFrame(purpose_variable,
                                    columns=['MEDV'])
    return explain_variable, purpose_variable


def standardization(data, mean=None, std=None):
    """
    Standardization
    If you do not specify mean and variance,
    circulate it.
    :param data: Data
    :param mean: Mean
    :param std: Variable
    :return: scalar, (mean, variance)
    """
    if mean is None and std is None:
        mean = np.mean(data)
        std = np.std(data)

        standard = (data - mean) / std
        return standard, mean, std
    else:
        standard = (data - mean) / std
        return standard


def compute_loss(ans, pred):
    """
    Mean Square Error
    :param ans: Answer
    :param pred: Predict
    :return: Loss
    """
    return mse(ans, pred)


def plot_graph(x, y, x_pred=None, y_pred=None, directory=None):
    """
    Plot Result

    :param x: train data (Explain variable)
    :param y: train data (Purpose Variable)
    :param x_pred: test data (Explain variable)
    :param y_pred: test data (Purpose variable)
    :param directory: save directory
    :return: None
    """
    sns.set()
    plt.scatter(x, y)
    if x_pred is not None and y_pred is not None:
        plt.scatter(x_pred, y_pred)
    else:
        pass

    plt.title('RM-MEDV')
    plt.xlabel('Explain Variance')
    plt.ylabel('Purpose Variance')
    plt.show()

    if directory is not None:
        plt.savefig(directory)
    else:
        pass


def lasso_return(alpha, test_size,  directory):
    """
    Main Processing of Lasso return
    :param alpha: lasso return parameter
    :param test_size: Ration of train and test
    :param directory: graph save params
    :return: None
    """
    ex_data, pp_data = load_data()

    ex_data = ex_data[['RM']]
    medv = pp_data['MEDV']

    # Spirit Train and Test
    ex_train, ex_test, medv_train, medv_test \
        = train_test_split(ex_data, medv,
                           test_size=test_size,
                           random_state=0)

    # Standardization
    ex_train_scalar, ex_mean, ex_std = \
        standardization(ex_train, mean=None, std=None)
    ex_test_scalar = standardization(data=ex_test, mean=ex_mean, std=ex_std)

    medv_train_scalar, m_mean, m_std = \
        standardization(medv_train, mean=None, std=None)
    medv_test_scalar = standardization(medv_test, mean=m_mean, std=m_std)

    # LASSO
    clf = Lasso(alpha=alpha)
    clf.fit(ex_train_scalar, medv_train_scalar)

    inclination = clf.coef_
    intercept = clf.intercept_
    print('{:15}'.format('prediction'), ' : ',
          'f(x) = ', '{:.4f}'.format(inclination[0]),
          'x + ', '{:.4f}'.format(intercept))
    print('************************************************')

    # Predict
    predict = clf.predict(ex_test_scalar)

    # computer LOSS
    print('{:15}'.format('mse'), ' : ',
          compute_loss(ans=medv_test_scalar, pred=predict))

    # Plot Graph
    plot_graph(x=ex_train_scalar,
               y=medv_train_scalar,
               x_pred=ex_test_scalar,
               y_pred=predict,
               directory=directory)


def main():
    """
    Main Function
    :return: None
    """
    # ------------------------------------
    # Params

    lasso_alpha = 0.1
    spirit_size = 0.2
    print('************************************************')
    print('{:15}'.format('Alpha'), ' : ', lasso_alpha)
    print('{:15}'.format('Spirit size'), ' : ', spirit_size)
    # ------------------------------------
    # Save Directory

    save_directory = None
    print('************************************************')
    print('{:15}'.format('Save Dir'), ' : ', save_directory)

    # ------------------------------------
    print('************************************************')
    lasso_return(alpha=lasso_alpha,
                 test_size=spirit_size,
                 directory=save_directory)
    print('************************************************')
    print('Done')
    print('************************************************')


if __name__ == '__main__':
    main()
