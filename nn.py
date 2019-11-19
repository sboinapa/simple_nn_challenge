import os
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.neural_network import MLPClassifier

# main_dir = "/Users/Sri/Desktop/data"
# os.chdir(main_dir)
# print(os.getcwd())

class NN:

	def __init__(self, train, test, nn_architecture=None):
		self.train, self.test, self.clf = train, test, None
        #setting architecture of neural network
		if not nn_architecture:
			self.nn_arch = (1600, 1600, 1600, 400, 200, 100)

	def train_nn(self):
        #load dataset file
        #skip the first row since its just the feature names
		dataset = np.loadtxt(self.train, delimiter=',', skiprows=1)
		np.random.shuffle(dataset)
        #split the set into training and validation
		train, val = dataset[:int(0.95*len(dataset))], dataset[int(0.95*len(dataset)):]

		train_X, train_y = train[:, 1:-1], train[:, -1]

        #set the classifier -- multilayer perceptron
		self.clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=self.nn_arch,
							random_state=1)
       
        #fitting the model on the set
		self.clf.fit(train_X, train_y)
        
        #use validation set to figure out the accuracy
		val_X = val[:, 1:-1]
		val_y = val[:, -1]
		score = self.clf.score(val_X, val_y)
		print(f'The score of the current classifier is {score}')

	def output_csv(self):
		test = np.loadtxt(self.test, delimiter=',', skiprows=1)
		test_X, ids = test[:, 1:], test[:, 0]
		results = np.vstack((ids, self.clf.predict(test_X))).T
		np.savetxt('output.csv', results, fmt='%i', delimiter=',')



Solution = NN("train.csv","test.csv")
Solution.train_nn()
Solution.output_csv()