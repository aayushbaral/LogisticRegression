import numpy as np
import scipy
import math

def calculateSigmoid(data):
	g  = 1.0/(1.0 + np.exp(-data))
	return g

def logistic_function(theta, x_values):
	data = np.dot(x_values, theta)
	return calculateSigmoid(data)




def cost_function(points, theta):
	
	x_points = points[:, 0:2]
	y_points = points[:, 2]
	first_expression = y_points * np.log(logistic_function(theta, x_points))
	second_expression = (1-y_points) * np.log(1 - logistic_function(theta, x_points))

	final_expression = (-first_expression) - second_expression

	return np.mean(final_expression)


def gradient_descent(initial_theta, learning_rate, num_iterations):
	theta = initial_theta
	for i in range(0, num_iterations):
		theta = step_gradient(theta, points, learning_rate)

	return theta

def step_gradient(theta_current, points, learning_rate):
	#x_points = points[:, 0:2]
	#y_points = points[:, 2]
	theta_current = np.zeros((2, 1))
	num = float(len(points))
	for i in range(0, len(points)):
		x_points = points[i, 1]

		y_points = points[i, 2]


		derivative_value = 1/num * (y_points - logistic_function(theta_current, x_points)) * x_points

		new_theta = theta_current  - learning_rate * derivative_value
	print x_points
	return new_theta
	


points = np.genfromtxt("C:\Users\click\Desktop\seeds_dataset.txt", usecols = [0, 1, 7])






point = points.copy()
point = point[:, 0:2]

initial_theta = np.zeros((2,1))
#print cost_function(points, initial_theta)
gradient_descent(initial_theta, 0.001, 10000)



# def gradient_descent(initial_theta, learning_rate, num_iterations):
# 	theta = initial_theta

# 		for i in range(num_iteration):
# 		#update b and m with new more accurate gradient descent

# 		theta = step_gradient(theta, points, learning_rate)

# 	return theta















# def run():
# 	points  = np.genfromtxt("C:\Users\click\Desktop\seeds_dataset.txt")

# 	#Defining the hyperparameters

# 	learning_rate = 0.0001

# 	inital_theta = np.zeros(1, 2)

# 	num_iterations = 1000

# 	print "Staarting gradient descent is "



# if __name__ == '__main__':
# 	run()