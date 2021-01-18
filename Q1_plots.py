from scratch import MyLinearRegression,MyPreProcessor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def kfoldCrossValidation(alpha,epochs,k,loss,dataset, plot=False):
    """
    Performing K-Fold Cross Validation

    Parameters
    ----------
    alpha : learning rate
    
    epochs : number of iterations in gradient descend
    
    k : k in k-fold
    
    loss: loss function to be used "RMSE" or "MAE"

    dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
    
    plot: True - plot avg train/test loss vs epochs
        False - do not plot
        default : False
    
    Returns
    -------
    tuple : (avg train loss, avg validation loss)

    Example:
    # alpha = 0.3
    # epochs = 250
    # k = 5
    # loss = "MAE"
    # dataset = 0
    # plot = True
    CrossValidation(0.3,250,5,"MAE",0,plot=True)
    """
    proc = MyPreProcessor()             #Initialize preprocessor
    X,y = proc.pre_process(dataset)     #Get the data

    #For avg loss plots
    avg_loss_train = np.zeros(epochs)
    avg_loss_test = np.zeros(epochs)

    for j in range(k):

        # Creating the splits into training and testing folds

        split = X.shape[0]//k

        Xtest = X[ split*j : split*(j+1), : ]
        ytest = y[ split*j : split*(j+1) ]

        Xtrain = None
        ytrain = None

        if j==0:        #corner case 1
            Xtrain = X[split:,:]
            ytrain = y[split:]
        elif j==k-1:    #corner case 2
            Xtrain = X[:split*(k-1),:]
            ytrain = y[:split*(k-1)]
        else:           #remaining cases
            Xtrain = X[ : split*j, :]
            np.append(Xtrain, X[ split*(j+1) :, : ], axis=0)
            ytrain = y[ : split*j]
            np.append(ytrain, y[ split*(j+1) : ], axis=0)

        linear = MyLinearRegression()   #Initialize linear regression model
        linear.fit(Xtrain, ytrain, plot=True, loss=loss, X_valid=Xtest, y_valid=ytest, alpha=alpha, epochs=epochs)  #Train the model using k-1 folds

        #Summing up corresponding losses at each epoch
        avg_loss_train += linear.train_loss
        avg_loss_test += linear.valid_loss

    #Divide by k for getting the avg
    avg_loss_train/=k
    avg_loss_test/=k

    if(plot):   #If asked, plot avg training and validation loss vs epochs
        proc.PlotLossVsEpochs(avg_loss_train, "Training Loss", avg_loss_test, "Validation Loss", text=str(k)+"-Fold Cross Validation; Dataset:"+str(dataset)+"; Loss:"+loss+"\nAvg Train Loss:"+str(avg_loss_train[-1])+"\nAvg Validation Loss:"+str(avg_loss_test[-1]))

    return avg_loss_train[-1],avg_loss_test[-1]     #return final avg train and validation loss

def findBestValueOfK():
    """
    Parameter: None
    Returns: None

    This is just a utility function for finding the value of k which achieves minimum loss in k-fold cross validation
    """
    per_plot = False
    #Accumulator variables
    min_k_train=[0]*4
    min_k_test=[0]*4
    min_loss_train=[float("inf")]*4
    min_loss_test=[float("inf")]*4

    for k in range(5,10+1): #Try for all 5<=k<=10
        # RMSE, Dataset-0
        RMSE_0 = kfoldCrossValidation(1,1000,k,"RMSE",0,plot=per_plot)
        if RMSE_0[0]<min_loss_train[0]:
            min_loss_train[0]=RMSE_0[0]
            min_k_train[0]=k
        if RMSE_0[1]<min_loss_test[0]:
            min_loss_test[0]=RMSE_0[1]
            min_k_test[0]=k
        # MAE, Dataset-0
        MAE_0 = kfoldCrossValidation(0.3,250,k,"MAE",0,plot=per_plot)
        if MAE_0[0]<min_loss_train[1]:
            min_loss_train[1]=MAE_0[0]
            min_k_train[1]=k
        if MAE_0[1]<min_loss_test[1]:
            min_loss_test[1]=MAE_0[1]
            min_k_test[1]=k
        # RMSE, Dataset-1
        RMSE_1 = kfoldCrossValidation(0.0003,1000,k,"RMSE",1,plot=per_plot)
        if RMSE_1[0]<min_loss_train[2]:
            min_loss_train[2]=RMSE_1[0]
            min_k_train[2]=k
        if RMSE_1[1]<min_loss_test[2]:
            min_loss_test[2]=RMSE_1[1]
            min_k_test[2]=k
        # MAE, Dataset-1
        MAE_1 = kfoldCrossValidation(0.00003,100,k,"MAE",1,plot=per_plot)
        if MAE_1[0]<min_loss_train[3]:
            min_loss_train[3]=MAE_1[0]
            min_k_train[3]=k
        if MAE_1[1]<min_loss_test[3]:
            min_loss_test[3]=MAE_1[1]
            min_k_test[3]=k
    #Print the results
    print("Order followed below: RMSE dataset0, MAE dataset0, RMSE dataset1, MAE dataset1")
    print("Values of K giving min loss | Calculated minimum loss for that K")
    print("Training:",min_k_train,min_loss_train)
    print("Validation:",min_k_test,min_loss_test)

def MSE_LOSS(X,y,theta):
    X_theta_minus_y = np.dot(X,theta) - y
    sq_sum = np.dot(X_theta_minus_y.T, X_theta_minus_y)
    return sq_sum/X.shape[0]

def kfoldNormalForm(dataset,k):
    """
    Performing K-Fold Cross Validation using Normal Form

    Parameters
    ----------
    
    k : k in k-fold
    
    dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
    
    Returns
    -------
    tuple : (avg train loss, avg validation loss)
    """
    # Get the preprocessed dataset
    proc = MyPreProcessor()
    X,y = proc.pre_process(dataset)
    X = np.insert(X,0,np.ones(X.shape[0]),axis=1)

    avg_train_loss = 0
    avg_val_loss = 0

    avg_optimal_theta = np.zeros(X.shape[1])

    for j in range(k):
        # Creating the splits into training and testing folds
        split = X.shape[0]//k

        Xtest = X[ split*j : split*(j+1), : ]
        ytest = y[ split*j : split*(j+1) ]

        Xtrain = None
        ytrain = None

        if j==0:        #corner case 1
            Xtrain = X[split:,:]
            ytrain = y[split:]
        elif j==k-1:    #corner case 2
            Xtrain = X[:split*(k-1),:]
            ytrain = y[:split*(k-1)]
        else:           #remaining cases
            Xtrain = X[ : split*j, :]
            np.append(Xtrain, X[ split*(j+1) :, : ], axis=0)
            ytrain = y[ : split*j]
            np.append(ytrain, y[ split*(j+1) : ], axis=0)

        # Calculate optimal parameters
        xtx = np.dot(Xtrain.T,Xtrain)
        xty = np.dot(Xtrain.T,ytrain)
        xtx_inv = np.linalg.pinv(xtx)
        theta = np.dot(xtx_inv, xty)

        avg_optimal_theta+=theta

        avg_train_loss += MSE_LOSS(Xtrain,ytrain,theta)
        avg_val_loss += MSE_LOSS(Xtest,ytest,theta)
    
    # Calculate average loss
    avg_train_loss/=k
    avg_val_loss/=k

    # Calculate average optimal theta
    avg_optimal_theta/=k

    print(avg_optimal_theta)

    return avg_train_loss,avg_val_loss

        
if __name__=="__main__":
    per_plot = True
    
    # Find which K gives min loss in all these cases
    findBestValueOfK()
    
    # Choose a k that happens to be good
    k = 8
    # RMSE, dataset-0
    RMSE_0 = kfoldCrossValidation(1,1000,k,"RMSE",0,plot=per_plot)
    # MAE, dataset-0
    MAE_0 = kfoldCrossValidation(0.3,250,k,"MAE",0,plot=per_plot)
    # RMSE, dataset-1
    RMSE_1 = kfoldCrossValidation(0.0003,1000,k,"RMSE",1,plot=per_plot)
    # MAE, dataset-1
    MAE_1 = kfoldCrossValidation(0.00003,100,k,"MAE",1,plot=per_plot)

    # k-fold cross validation using normal form
    print(kfoldNormalForm(0,k))
