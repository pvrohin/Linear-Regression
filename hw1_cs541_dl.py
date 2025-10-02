# -*- coding: utf-8 -*-
"""HW1-CS541-DL.py

Author : Rohin Siddhartha Palaniappan Venkateswaran
Student ID: 808068806

Colab file is located at
    https://colab.research.google.com/drive/1TFz5FjcRDnlaV0DlD698rklgmUoKVPR1#scrollTo=FAiiHXas2gox
"""

#import required libraries
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import sys

#Function to add 2 matrices A and B as per question 1.(a)
def problem_1a (A, B):
    return A + B

#Function to compute AB - C as per question 1.(b), array multiplication operator @ is used here as per instructions from the question.
def problem_1b (A, B, C):
    return A@B - C

#Function to compute A⊙B + C⊤ as per question 1.(c)
def problem_1c (A, B, C):
    return A*B+ C.T

#Function to compute the inner product of x and y
def problem_1d (x, y):
    return x.T@y

#Function to compute A −1 x
def problem_1e (A, x):
    return np.linalg.solve(A,x)

#Function to compute xA −1
def problem_1f (A, x):
    return (np.linalg.solve(A.T, x.T)).T

#Function to return the sum of all the entries in the ith row whose column index is even
def problem_1g (A, i):
    return np.sum(A[i,0::2]) #step here is given as 2 since we need to extract even index elements, so we need to skip over the odd elements

#Function to compute the arithmetic mean over all entries of A that are between c and d (inclusive)
def problem_1h (A, c, d):
    np.where(A >= c, 0, A) # change all the elements in A which are >= c to zero.
    np.where(A <= d, 0, A) # change all the elements in A which are <= d to zero.
    return np.mean(A[np.nonzero(A)]) # calculate the mean of the rest of the elements which are non-zero

#Function to return an (n×k) matrix containing the right-eigenvectors of A corresponding to the k largest eigenvalues of A
def problem_1i (A, k):
    eigen_value, eigen_vector = np.linalg.eig(A) # obtain the eigen values and eigen vector from A using np.linalg.eig
    n = len(eigen_value) # compute the length of eigen value vector
    descen_arr_indices = (eigen_value).argsort()[:n] # arrange all the indices of eigen value vector in decreasing order
    size = len(descen_arr_indices) # compute the length of the new vector of decreasing indices
    matching_indices = descen_arr_indices[size - k:size] # remove the eigen value indices which are not in the k largest eigen values of A
    return eigen_vector[:,matching_indices] # return the respective eigen vectors which only correspond to the k largest eigen values of A

# Function to return an (n × k) matrix, each of whose columns is a sample from multidimensional Gaussian distribution
def problem_1j (x, k, m, s):
    n = len(x) # Calculate the length of x
    z = np.ones((n,1)) # Initialize z as a vector of ones of size n x 1
    m_z = m * z # Calculate mz
    I = np.eye(n) # Initialize an identity matrix of size n x n
    s_I = s * I # Calculate sI
    return np.random.multivariate_normal(np.add(x, m_z).squeeze(), s_I, k).T # Calculate the multi-dimensional Gaussian distribution matrix

# Function to return a matrix that results from randomly permuting the rows (but not the columns) in A
def problem_1k (A):
    row_size = A.shape[0] # Calculate the dimension of the row size
    return A[np.random.permutation(row_size),:] # Return A by randomly permuting the rows only and not columns

# Function to return the z-score
def problem_1l (x):
    return (x-np.mean(x))/np.std(x) # Straight-forward implementatio of Z-scoring

# Function to return a n × k matrix consisting of k copies of x.
def problem_1m (x, k):
    x_2d = np.repeat(x, k, axis=0) # Repeat the x matrix k times.
    return x_2d

# Function to compute a matrix which has all pairwise L2 distances
def problem_1n (X):
  width = X.shape[1] # Extract the wodth of X
  Y = np.repeat(np.atleast_3d(X),width,axis = 2) # Convert the 2D model into a 3 model
  Y_swap = np.swapaxes(Y,1,2) # Swap axis of Y
  L2_dist = ((Y_swap-Y)**2).sum(axis=0)**.5 # Compute L2 distance
  return pow(L2_dist,.5) #Return L2 distance

#Function to perform linear regression
def linear_regression (X_tr, y_tr):
    first_term = np.dot(X_tr.T,X_tr)
    second_term = np.dot(X_tr.T,y_tr)
    return np.linalg.solve(first_term, second_term)

# Loss function to calculate the MSE Loss
def loss_fn(y_h,y):
    return np.mean((y_h - y)**2)/2

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("/home/rohin/Downloads/age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("/home/rohin/Downloads/age_regression_ytr.npy")
    X_te = np.reshape(np.load("/home/rohin/Downloads/age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("/home/rohin/Downloads/age_regression_yte.npy")

    # Obtain w by calling the linear_regression function
    w = linear_regression(X_tr, ytr)

    # Obtain the output from the training set
    training_output = np.dot(X_tr,w)

    # Obtain the output from the testing set
    testing_output = np.dot(X_te,w)

    # Call the loss function to obtain training and testing losses respectively
    train_error = loss_fn(training_output, ytr)
    test_error = loss_fn(testing_output, yte)

    return train_error, test_error

#Function to compare consistency of given data with alternative rate parameter 2.5
def rp_x1(x1,data):
  plt.hist(data,density=True)
  plt.hist(x1, density=True, edgecolor='black')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("Rate Parameter 2.5")
  plt.show()

#Function to compare consistency of given data with alternative rate parameter 3.1
def rp_x2(x2,data):
  plt.hist(data,density=True)
  plt.hist(x2, density=True, edgecolor='black')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("Rate Parameter 3.1")
  plt.show()

#Function to compare consistency of given data with alternative rate parameter 3.7
def rp_x3(x3,data):
  plt.hist(data,density=True)
  plt.hist(x3, density=True, edgecolor='black')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("Rate Parameter 3.7")
  plt.show()

#Function to compare consistency of given data with alternative rate parameter 3.3
def rp_x4(x4,data):
  plt.hist(data,density=True)
  plt.hist(x4, density=True, edgecolor='black')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title("Rate Parameter 3.3")
  plt.show()

def main():

    #Declare A,B and C matrices here
    A = np.array([[5,27,31],
    [3,11,23],
    [1,17,20]])

    B = np.array([[12,30,24],
    [7,13,19],
    [0,4,18]])

    C = np.array([[10,15,2],
    [8,16,21],
    [6,25,29]])

    # Declare separate row and column matrices for x and y for ease of use.

    # x column vector
    x_c = np.array([[35],
    [52],
    [12]])

    # x row vector
    x_r = np.array([[35,52,12]])

    # y column vector
    y_c = np.array([[72],
    [56],
    [0],
    ])

    # y row vector
    y_r = np.array([[72,56,0]])

    # X matrix
    X = np.array([[10,20,30],[40,50,60]])

    #initialize the scalars
    i = 2
    c = 1
    d = 20
    k = 2
    m = 3
    s = 7
    k = 5
    k = 5

    #Functions 1a to 1n
    print("Output of problem_1a: ",problem_1a(A,B))
    print("Output of problem_1b: ",problem_1b(A,B,C))
    print("Output of problem_1c: ",problem_1c(A,B,C))
    print("Output of problem_1d: ",problem_1d(x_c,y_c))
    print("Output of problem_1e: ",problem_1e(A,x_c))
    print("Output of problem_1f: ",problem_1f(A,x_r))
    print("Output of problem_1g: ",problem_1g(A,i))
    print("Output of problem_1h: ",problem_1h(A,c,d))
    print("Output of problem_1i: ",problem_1i(C,k))
    print("Output of problem_1j: ",problem_1j(x_c,k,m,s))
    print("Output of problem_1k: ",problem_1k(A))
    print("Output of problem_1l: ",problem_1l(x_c))
    print("Output of problem_1m: ",problem_1m(x_r,k))
    print("Output of problem_1n: ",problem_1n(X))

    #2.(a) Linear Regression via Analytical Solution

    print("Training and testing error are: ",train_age_regressor())

    #3.(a) Probability Distributions
    data = np.load("/home/rohin/Downloads/PoissonX.npy") # Load the data

    plt.hist(data,density=True) # Plot the empirical probablility distribution given PoissonX.npy data

    plt.xlabel('x')
    plt.ylabel('y')

    # displaying the title
    plt.title("PoissonX.npy")

    plt.show()

    input("Press Enter to continue...")

    #generate Poisson distribution with alternative rate parameter of 2.5
    x1 = poisson.rvs(mu=2.5, size=10000)

    #generate Poisson distribution with alternative rate parameter of 3.1
    x2 = poisson.rvs(mu=3.1, size=10000)

    #generate Poisson distribution with alternative rate parameter of 3.7
    x3 = poisson.rvs(mu=3.7, size=10000)

    #generate Poisson distribution with alternative rate parameter of 3.3
    x4 = poisson.rvs(mu=3.3, size=10000)

    #create plot of Poisson distributions generated above
    rp_x1(x1,data)

    input("Press Enter to continue...")

    rp_x2(x2,data)

    input("Press Enter to continue...")

    rp_x3(x3,data)

    input("Press Enter to continue...")

    rp_x4(x4,data)

    sys.exit("End of program : Press Ctrl+C to exit")

main()
