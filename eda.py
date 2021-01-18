""" 
Exploratory Data Analysis Plot for Bank Note dataset - 2

Produces a plot for all the features

Prints description of the dataset to the console
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Datasets\\BankNote.txt",header=None)

print(data.describe())

print("Labelled 0:",np.sum((data[4]==0)*1))
print("Labelled 1:",np.sum((data[4]==1)*1))

plt.plot(data[0], label="variance of Wavelet Transformed image")
plt.plot(data[1], label="skewness of Wavelet Transformed image")
plt.plot(data[2], label="curtosis of Wavelet Transformed image")
plt.plot(data[3], label="entropy of image")
plt.legend()
plt.suptitle("Banknote Authentication Data Set")
plt.show()