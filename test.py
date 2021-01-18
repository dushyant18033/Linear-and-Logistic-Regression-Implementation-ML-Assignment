from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np

preprocessor = MyPreProcessor()

print('Linear Regression')

X, y = preprocessor.pre_process(0)

# Create your k-fold splits or train-val-test splits as required

split = X.shape[0]//5

Xtrain = X[split:]
ytrain = y[split:]
Xtest = X[:split]
ytest = y[:split]

linear = MyLinearRegression()
linear.fit(Xtrain, ytrain)

ypred = linear.predict(Xtest)

print('Predicted Values:', ypred)
print('True Values:', ytest)

print('Logistic Regression')

X, y = preprocessor.pre_process(2)

# Create your k-fold splits or train-val-test splits as required
split = X.shape[0]//5

Xtrain = X[split:]
ytrain = y[split:]
Xtest = X[:split]
ytest = y[:split]

logistic = MyLogisticRegression()
logistic.fit(Xtrain, ytrain)

ypred = logistic.predict(Xtest)

print('Predicted Values:', ypred)
print('True Values:', ytest)