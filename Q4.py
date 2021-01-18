from scratch import MyLogisticRegression,MyPreProcessor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Datasets\\Q4_Dataset.txt", header=None, delimiter=r"\s+")

X = data[[1,2]]
y = data[0]

X=X.to_numpy()      #Converting to numpy array
y=y.to_numpy()      #Converting to numpy array

model = MyLogisticRegression()

# Training model
model.fit(X, y, alpha=0.003, epochs=100000, plot=True, X_valid=X, y_valid=y)

# Estimated parameters for Q4.a part
print("Parameters:",model.theta)

# Reporting accuracy
acc = np.sum((model.predict(X)==y)*1)/X.shape[0]
print(acc)

# Prediction for Q4.d part
print("Predictions:",model.predict_prob(np.array([[75,2]])))

# # Plot loss vs epoch
# plt.plot(list(range(len(model.train_loss))),model.train_loss)
# plt.xlabel('Epochs/Iterations')
# plt.ylabel('Loss')
# plt.show()
