from scratch import MyLogisticRegression,MyPreProcessor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression,SGDClassifier



def resultsUsingSKLogistic():
    """
    Perform Logistic Regression on dataset-2 using scikit learn's built-in methods

    Reports Train and test accuracy

    Parameters: None

    Returns: None
    """
    print("Using SkLearn Logistic...")

    # Obtain the data
    proc = MyPreProcessor()
    X,y = proc.pre_process(2)
    
    k = X.shape[0]//10

    # Train split
    Xtrain = X[:7*k, :]
    ytrain = y[:7*k]
    # Validation split
    Xvalid = X[7*k:8*k, :]
    yvalid = y[7*k:8*k]
    # Test split
    Xtest = X[8*k:, :]
    ytest = y[8*k:]

    # Initialize the model
    logistic = LogisticRegression()
    
    # Train the model
    logistic.fit(Xtrain,ytrain)

    # Calculate accuracy
    train_acc = logistic.score(Xtrain,ytrain)
    test_acc = logistic.score(Xtest,ytest)

    # Report the results
    print("Train Accuracy:",train_acc)
    print("Test Accuracy:",test_acc)

def resultsUsingSGDClassifier():
    """
    Perform SGD Logistic Regression on dataset-2 using scikit learn's built-in methods

    Reports Train and test accuracy

    Parameters: None

    Returns: None
    """
    print("Using SkLearn SGD Classifier...")

    # Obtain the data
    proc = MyPreProcessor()
    X,y = proc.pre_process(2)
    
    k = X.shape[0]//10

    # Train split
    Xtrain = X[:7*k, :]
    ytrain = y[:7*k]
    # Validation split
    Xvalid = X[7*k:8*k, :]
    yvalid = y[7*k:8*k]
    # Test split
    Xtest = X[8*k:, :]
    ytest = y[8*k:]

    # Initialize the model
    model = SGDClassifier(loss='log', max_iter=100000, learning_rate='constant', eta0=0.1)
    
    # Train the model
    model.fit(Xtrain,ytrain)

    # Calculate accuracy
    train_acc = model.score(Xtrain,ytrain)
    test_acc = model.score(Xtest,ytest)

    # Report the results
    print("Train Accuracy:",train_acc)
    print("Test Accuracy:",test_acc)

def resultsUsingMyLogistic(alg,alpha,epochs):
    """
    Performing logistic regression on dataset-2 using alg type gradient descent

    
    Parameters
    ----------
    alpha : learning rate
    
    epochs : number of iterations in gradient descend
    
    alg: "SGD" or "BGD"
    ----------

    Returns: None
    """

    print("Using MyLogistic...")
    
    # Importing the data
    proc = MyPreProcessor()
    X,y = proc.pre_process(2)
    
    k = X.shape[0]//10

    # Train split
    Xtrain = X[:7*k, :]
    ytrain = y[:7*k]
    # Valid split
    Xvalid = X[7*k:8*k, :]
    yvalid = y[7*k:8*k]
    # Test split
    Xtest = X[8*k:, :]
    ytest = y[8*k:]

    # Initialize my own implementation of Logistic Regression
    logistic = MyLogisticRegression()
    logistic.fit(Xtrain, ytrain, plot=True, X_valid=Xvalid, y_valid=yvalid, alpha=alpha, epochs=epochs, alg=alg)

    # Print the results
    print("Final Train Loss:",logistic.train_loss[-1])
    print("Final Validation Loss:",logistic.valid_loss[-1])
    print("Theta:",logistic.theta)

    train_acc = np.sum((logistic.predict(Xtrain)==ytrain)*1)/Xtrain.shape[0]
    test_acc = np.sum((logistic.predict(Xtest)==ytest)*1)/Xtest.shape[0]

    print("Train Accuracy:",train_acc)
    print("Test Accuracy:",test_acc)

    # Plot results
    proc.PlotLossVsEpochs(logistic.train_loss, "Training Loss", logistic.valid_loss, "Validation Loss", text=alg+"; dataset-2"+"\nTrain Acc:"+str(train_acc)+"\nTest Acc:"+str(test_acc))


if __name__=="__main__":
    resultsUsingMyLogistic("SGD",0.1,100000)
    resultsUsingMyLogistic("BGD",3,10000)
    resultsUsingSKLogistic() #c | default logistic regression implementation
    resultsUsingSGDClassifier() #d | sgd classifier with attributes to make it behave as logistic regression with our hyper parameters

    # resultsUsingMyLogistic("SGD",10,100000)
    # resultsUsingMyLogistic("BGD",10,10000)