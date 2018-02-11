#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot

# a = np.random.rand(2,2)
# P = a/a.sum()

P = np.matrix([[0.4, 0.6], 
			   [0.3, 0.7]])

# Get the data
plot_data = []
for step in range(10):
    result = np.eye(1,2) * P**step
    plot_data.append(np.array(result).flatten())

# Convert the data format
plot_data = np.array(plot_data)

print(plot_data)

# Create the plot
pyplot.figure(1)
pyplot.xlabel('Steps')
pyplot.ylabel('Probability')
pyplot.plot(plot_data)
pyplot.show()