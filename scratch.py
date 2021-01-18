import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def PlotLossVsEpochs(self, plot1, label1, plot2, label2, text=None):
        epochs = len(plot1)
        x_axis = list(range(epochs))
        plt.plot(x_axis,plot1, label=label1)
        plt.plot(x_axis,plot2, label=label2)
        plt.legend()
        plt.xlabel('Epochs/Iterations')
        plt.ylabel('Loss')
        if text is not None:
            plt.suptitle(str(text))
        plt.show()

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        # np.empty creates an empty array only. You have to replace this with your code.
        

        if dataset == 0:
            abalone = pd.read_csv("Datasets\\AbaloneDataset.data", header=None, delimiter=r"\s+")

            #Binary profiling for Sex: M,F,I
            abalone.insert(0,"Male",(abalone[0]=="M")*1,True)
            abalone.insert(0,"Female",(abalone[0]=="F")*1,True)
            abalone.insert(0,"Infant",(abalone[0]=="I")*1,True)
            abalone.drop(columns=[0],inplace=True)

            abalone = abalone.sample(frac=1, random_state=42)    #Shuffling the data

            X=abalone[["Male","Female","Infant",1,2,3,4,5,6,7]] #Extracting input variables
            y=abalone[8]                                        #Extracting output variable

            X=X.to_numpy()      #Converting to numpy array
            y=y.to_numpy()      #Converting to numpy array

            # X = X / X.max(axis=0)  #normalizing wrt max element, data will be between 0 and 1
            # y = y / y.max(axis=0)  #normalizing wrt max element, data will be between 0 and 1

            return X,y

        elif dataset == 1:
            game = pd.read_csv("Datasets\\VideoGameDataset.csv", usecols=["Critic_Score","User_Score","Global_Sales"])

            game.dropna(axis=0, how="any", inplace=True)    #Removing rows with NaN or None data
            game = game[ game["User_Score"]!="tbd" ]        #Removing rows containing 'tba' as data
            game = game.sample(frac=1, random_state=42)     #Shuffling the data contents

            X=game[["Critic_Score","User_Score"]]   #Extracting input variables
            y=game["Global_Sales"]                #Extracting output variable

            X=X.to_numpy()      #Converting to numpy array
            y=y.to_numpy()      #Converting to numpy array
            X = X.astype(float)   #Converting string to numeric type

            # X = X / X.max(axis=0)  #normalizing wrt max element, data will be between 0 and 1
            # y = y / y.max(axis=0)  #normalizing wrt max element, data will be between 0 and 1

            return X,y            
            
        elif dataset == 2:
            bank = pd.read_csv("Datasets\\BankNote.txt", header=None)
            bank = bank.sample(frac=1, random_state=42)    #Shuffling the data

            X=bank[[0,1,2,3]]   #Extracting input variables
            y=bank[4]           #Extracting output variables

            X=X.to_numpy()      #Converting to numpy array
            y=y.to_numpy()      #Converting to numpy array

            X = X / X.max(axis=0)  #normalizing wrt max element, data will be between 0 and 1
            y = y / y.max(axis=0)  #normalizing wrt max element, data will be between 0 and 1

            return X,y

        else:
            X = np.empty((0,0))
            y = np.empty((0))
            return X, y


class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """

    def __init__(self):
        pass

    def fit(self, X, y, loss="RMSE", alpha=0.01, epochs=100, plot=False, X_valid=None, y_valid=None):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        X = np.insert(X,0,np.ones(X.shape[0]),axis=1)   # adding a column of ones
        if(X_valid is not None):
            X_valid = np.insert(X_valid,0,np.ones(X_valid.shape[0]),axis=1)

        def J_theta(X,y,theta,loss=loss):       # Loss function
            if loss == "RMSE":
                X_theta_minus_y = np.dot(X,theta) - y
                sq_sum = np.dot(X_theta_minus_y.T, X_theta_minus_y)
                return sqrt(sq_sum/X.shape[0])
            elif loss == "MAE":
                X_theta_minus_y = np.dot(X,theta) - y
                return np.sum(np.fabs(X_theta_minus_y)) / X.shape[0]
            elif loss == "MSE":
                X_theta_minus_y = np.dot(X,theta) - y
                sq_sum = np.dot(X_theta_minus_y.T, X_theta_minus_y)
                return sq_sum/X.shape[0]

        def slope_J_theta(X,y,theta,loss=loss): # Gradient function
            if loss == "RMSE":
                X_theta_minus_y = np.dot(X,theta) - y
                sq_sum = np.dot(X_theta_minus_y.T, X_theta_minus_y)
                j_theta =  sqrt(sq_sum/X.shape[0])
                return np.dot(X_theta_minus_y.T, X) / (j_theta*X.shape[0])
            elif loss == "MAE":
                signum = np.sign(np.dot(X,theta) - y)
                return np.dot(signum.T,X)/X.shape[0]
            elif loss == "MSE":
                X_theta_minus_y = np.dot(X,theta) - y
                temp = np.dot(X_theta_minus_y.T, X)
                return 2*temp/X.shape[0]
        
        self.theta = [0]*X.shape[1]     # initializing parameters to zero

        if(plot):
            self.train_loss = list()
            self.valid_loss = list()

        to_plot = loss # "MSE"  # set this to MSE to view MSE based performance

        # Gradient Descent algorithm
        for i in range(epochs): 
            self.theta = self.theta - alpha*slope_J_theta(X,y,self.theta,loss=loss)
            if plot and (X_valid is not None) and (y_valid is not None):
                self.train_loss.append(J_theta(X,y,self.theta,loss=to_plot))
                self.valid_loss.append(J_theta(X_valid,y_valid,self.theta,loss=to_plot))
        
        return self

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        X = np.insert(X,0,np.ones(X.shape[0]),axis=1)
        return np.dot(X,self.theta)


class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self):
        pass

    
    
    def fit(self, X, y, alpha=0.01, epochs=100, plot=False, X_valid=None, y_valid=None, alg="BGD"):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        X = np.insert(X,0,np.ones(X.shape[0]),axis=1)   # Adding column of ones
        if(X_valid is not None):
            X_valid = np.insert(X_valid,0,np.ones(X_valid.shape[0]),axis=1)

        def sigmoid(z):     # Returns sigmoid applied to all elements of any numpy array
            return 1.0/(1.0 + np.exp(-z))

        def J_theta(X,y,theta):     # Loss function
            h_theta = sigmoid(np.dot(X,theta))
            # h_theta += (h_theta==1)*0.00000001 + (h_theta==0)*0.00000001
            part1 = np.dot( np.log(h_theta), y )
            part2 = np.dot( np.log(1-h_theta), 1-y )
            return -(part1+part2)/X.shape[0]
        
        def slope_J_theta(X,y,theta, alg="BGD"):    # Gradient function
            if alg == "SGD":
                i = np.random.randint(X.shape[0])
                X_theta_minus_y = sigmoid(np.dot(X[i],theta)) - y[i]
                return X_theta_minus_y*X[i]
            else:
                X_theta_minus_y = sigmoid(np.dot(X,theta)) - y
                return np.dot(X_theta_minus_y.T,X)/X.shape[0]


        self.theta = [0]*X.shape[1]     # Init parameters
        
        self.train_loss = list()
        self.valid_loss = list()

        # Gradient Descend in action
        for i in range(epochs):
            self.theta = self.theta - alpha*slope_J_theta(X,y,self.theta, alg=alg)
            if plot and (X_valid is not None) and (y_valid is not None):
                self.train_loss.append(J_theta(X,y,self.theta))
                self.valid_loss.append(J_theta(X_valid,y_valid,self.theta))
        
        return self

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        def sigmoid(z):
            return 1/(1 + np.exp(-z))
        
        X = np.insert(X,0,np.ones(X.shape[0]),axis=1)
        return (sigmoid(np.dot(X,self.theta))>=0.5)*1
    
    def predict_prob(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        def sigmoid(z):
            return 1/(1 + np.exp(-z))
        
        X = np.insert(X,0,np.ones(X.shape[0]),axis=1)
        return sigmoid(np.dot(X,self.theta))
