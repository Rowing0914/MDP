#!/usr/bin/env python
import numpy as np
import itertools
from matplotlib import pyplot
import random

class markov:
	def __init__(self, P, S, gamma):
		self.P = P
		self.S = S
		self.gamma = gamma

	# create actions => 3x8 matrix i.e. steps times (2**3)
	def make_action(self, Action):
		return list(itertools.product(Action, Action, Action))

	# state-value function => sum over (gamma**k)*(R of t+k+1) where k starts from 0
	def equation(self, i, weight):
		return (self.gamma**i)*weight[i]

	# calculate weights for each action, and aggregate them
	def value(self, Actions, State, gamma):
		values = []
		# temporarily store weight
		weight = []
		# store the weights
		weights = []
		total = 0

		for action in Actions:
			for a in action:
				# refer to get the proper reward at the state
				weight.append(State[a[0]])
			weights.append(weight)

			# calculate value with weights and gamma 
			for i in range(len(action)):
				total += self.equation(i, weight)

			values.append(total)

			# reset weight
			weight = []
		return values, weights

	# plot data
	def plot(self, plot_data):
		pyplot.figure(1)
		pyplot.xlabel('Steps')
		pyplot.ylabel('Probability')
		pyplot.plot(plot_data)
		pyplot.show()

if __name__ == '__main__':
	"""
	P: trainsition matrix
	S: States and rewards
	A: actions
	"""
	P = np.array([[0.4, 0.6], [0.3, 0.7]])
	S = {'A': -2, 'B':-1}
	A = np.array([['A'],['B']])
	gamma = 0.9

	# initialise markov
	markov = markov(P=P, S=S, gamma=gamma)
	Actions = markov.make_action(A)
	values, weights = markov.value(Actions, S, gamma=0.9)
	print(Actions)
	print(weights)
	print(values)

	# Get the data
	plot_data = []
	for step in range(10):
	    result = np.eye(1,2) * P**step
	    plot_data.append(np.array(result).flatten())

	# Convert the data format
	plot_data = np.array(plot_data)

	# markov.plot(plot_data)