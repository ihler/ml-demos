################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np
import random

from classify import Classify
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from numpy import concatenate as concat
from numpy import column_stack as cols
from utils.data import bootstrap_data, load_data_from_csv, rescale, split_data
from utils.utils import from_1_of_k, to_1_of_k


################################################################################
################################################################################
################################################################################


################################################################################
## NNETCLASSIFY ################################################################
################################################################################

def _add1(X):
	return np.hstack( (np.ones((X.shape[0],1)),X) )

class nnetClassify(classifier):
  """A simple neural network classifier
  Attributes:
    classes: list of class (target) identifiers for the classifier
    layers : list of layer sizes [N,S1,S2,...,C], where N = # of input features, S1 = # of hidden nodes 
             in layer 1, ... , and C = the number of classes, or 1 for a binary classifier
    weights: list of numpy arrays containing each layer's weights, size e.g. (S1,N), (S2,S1), etc.

  """

	def __init__(self, *args, **kwargs): #X, Y, sizes, init='zeros', stepsize=.01, tolerance=1e-4, max_steps=5000, activation='logistic'):
		"""
		Constructor for NNetClassifier (neural net classifier).

    Parameters: see the "train" function; calls "train" if arguments passed

    Properties:
      classes : list of identifiers for each class
      wts     : list of coefficients (weights) for each layer of the NN
      activation : function for layer activation function & derivative
		"""
		self.classes = []
		self.wts = arr([], dtype=object)
		#self.set_activation(activation.lower())
		#self.init_weights(sizes, init.lower(), X, Y)

		if len(args) or len(kwargs):     # if we were given optional arguments,
			self.train(*args, **kwargs)    #  just pass them through to "train"


	def __repr__(self):
		to_return = 'Multi-layer perceptron (neural network) classifier\nLayers [{}]'.format(self.get_layers())
		return to_return


	def __str__(self):
		to_return = 'Multi-layer perceptron (neural network) classifier\nLayers [{}]'.format(self.get_layers())
		return to_return

  def nLayers(self):
    return len(self.wts)

  @property
  def layers(self):
    if len(self.wts):
      layers = [self.wts[l].shape[1] for l in range(len(self.wts))]
      layers.append( self.wts[-1].shape[0] )
    else:
      layers = []
    return layers

  @layers.setter
  def layers(self, layers):
    # adapt / change size of weight matrices (?)


## TEMPORARY CODE ##############################################################
  Sig = lambda Z: np.tanh(Z)
  dSig= lambda Z: np.tanh(Z)**2


## CORE METHODS ################################################################

	def predictSoft(self, X):
		"""
		Make 'soft' (per-class confidence) predictions of the neural network on data X.
		"""
    X = arr(X)                               # convert to numpy if needed
		L = self.nLayers()                       # get number of layers
		Z = _add1(X)                             # initialize: input features + constant term

		for l in range(L - 2):                   # for all *except output* layer:
			Z = mat(Z) * mat(self.wts[l]).T        # compute linear response of next layer
      Z = _add1( self.Sig(Z) )               # apply activation function & add constant term

		Z = mat(Z) * mat(self.wts[L - 1]).T      # compute output layer linear response
		return self.Sig0(Z)                      # apply output layer activation function
    ### TODO: 1-p,p if Z is single-column?


	def train(self, X, Y, init='zeros', stepsize=.01, tolerance=1e-4, max_steps=5000):
		"""Train the neural network.

		Parameters
		----------
		X : numpy array
			N x M array that contains N data points with M features.
		Y : numpy array
			Array taht contains class labels that correspond
		  	to the data points in X. 
		sizes : [Nin, Nh1, ... , Nout] 
			Nin is the number of features, Nout is the number of outputs, 
			which is the number of classes. Member weights are {W1, ... , WL-1},
		  	where W1 is Nh1 x Nin, etc.
		init : str 
			'none', 'zeros', or 'random'.  inits the neural net weights.
		stepsize : scalar
			The stepsize for gradient descent (decreases as 1 / iter).
		tolerance : scalar 
			Tolerance for stopping criterion.
		max_steps : int 
			The maximum number of steps before stopping. 
		activation : str 
			'logistic', 'htangent', or 'custom'. Sets the activation functions.
		
		"""
		if self.wts[0].shape[1] - 1 != len(X[0]):
			raise ValueError('layer[0] must equal the number of columns of X (number of features)')

		if len(np.unique(Y)) != self.wts[-1].shape[0]:
			raise ValueError('layers[-1] must equal the number of classes in Y')

		self.classes = self.classes if len(self.classes) else np.unique(Y)

		# convert Y to 1-of-K format
		Y_tr_k = to_1_of_k(Y)

		M,N = mat(X).shape													# d = dim of data, n = number of data points
		C = len(self.classes)												# number of classes
		L = len(self.wts) 													# get number of layers

		# outer loop of stochastic gradient descent
		it = 1															# iteration number
		done = 0														# end of loop flag
		J01, Jsur = [],[] 							# misclassification rate & surrogate loss values

		while not done:
			step_i = stepsize / it										# step size evolution; classic 1/t decrease
			
			# stochastic gradient update (one pass)
			for i in range(n):
				A,Z = self.__responses(X[i,:])		# compute all layers' responses, then backdrop
				delta = (Z[L] - Y_tr_k[i,:]) * arr(self.dSig0(Z[L]))			# take derivative of output layer

				for l in range(L - 1, -1, -1):
					grad = delta.T.dot( Z[l] )							# compute gradient on current layer wts
					delta = delta.dot(self.wts[l]) * self.dSig(Z[l]) # propagate gradient downards
					delta = delta[:,1:]										# discard constant feature
					self.wts[l] -= step_i * grad				# take gradient step on current layer wts

			J01.append(  self.err_k(X, Y_tr_k) )								# error rate (classification)
			Jsur.append( self.mse_k(X, Y_tr_k) )								# surrogate (mse on output)

			print('it {} : Jsur = {}, J01 = {}'.format(it,Jsur[-1],J01[-1]))

			# check if finished
			done = (iter > 1) and (np.abs(surr[-1] - surr[-2]) < tolerance) or iter >= max_steps
			iter += 1




	def err_k(self, X, Y):
		"""
		Compute misclassification error. Assumes Y in 1-of-k form.
		See constructor doc string for argument descriptions.
		"""
		Y_hat = self.predict(X)
		return np.mean(Y_hat != from_1_of_k(Y))
	def log_likelihood(self, X, Y):
		"""
		Compute the emperical avg. log likelihood of 'obj' on test data (X,Y).
		See constructor doc string for argument descriptions.
		"""
		r,c = twod(Y).shape
		if r == 1 and c != 1:
			Y = twod(Y).T

		soft = self.predict_soft(X)
		return np.mean(np.sum(np.log(np.power(soft, Y, )), 1), 0)


	def mse(self, X, Y):
		"""
		Compute mean squared error of predictor 'obj' on test data (X,Y).
		See constructor doc string for argument descriptions.
		"""
		return mse_k(X, to_1_of_K(Y))


	def mse_k(self, X, Y):
		"""
		Compute mean squared error of predictor; assumes Y is
		in 1-of-k format. Refer to constructor docstring for
		argument descriptions.
		"""
		return np.power(Y - self.predict_soft(X), 2).sum(1).mean(0)


## MUTATORS ####################################################################


	#def set_activation(self, method, sig=None, d_sig=None, sig_0=None, d_sig_0=None):
	def setActivation(self, method, sig=None, sig0=None): 
		"""
		This method sets the activation functions. 

		Parameters
		----------
		method : string, {'logistic' , 'htangent', 'custom'} -- which activation type
    Optional arguments for "custom" activation:
    sig : function object F(z) returns activation function & its derivative at z (as a tuple)
    sig0: activation function object F(z) for final layer of the nnet
		"""
		method = method.lower()

		if method == 'logistic':
			self.sig = lambda z: twod(1 / (1 + np.exp(-z)))
			self.d_sig = lambda z: twod(np.multiply(self.sig(z), (1 - self.sig(z))))
			self.sig_0 = self.sig
			self.d_sig_0 = self.d_sig
		elif method == 'htangent':
			self.sig = lambda z: twod(np.tanh(z))
			self.d_sig = lambda z: twod(1 - np.power(np.tanh(z), 2))
			self.sig_0 = self.sig
			self.d_sig_0 = self.d_sig
		elif method == 'custom':
			self.sig = sig
			self.d_sig = d_sig
			self.sig_0 = sig_0
			self.d_sig_0 = d_sig_0
		else:
			raise ValueError('NNetClassify.set_activation: ' + str(method) + ' is not a valid option for method')

		self.activation = method



	def set_layers(self, sizes, init='random'):
		"""
		Set layers sizes to sizes.

		Parameters
		----------
		sizes : [int]
			List containing sizes.
		init : str (optional)
			Weight initialization method.
		"""
		self.init_weights(sizes, init, None, None)


	def init_weights(self, sizes, init, X, Y):
		"""
		This method initializes the weights of the neural network and
		sets layer sizes to S=[Ninput, N1, N2, ... , Noutput]. Refer
		to constructor doc string for descritpions of arguments.
		"""
		init = init.lower()

		if init == 'none':
			pass
		elif init == 'zeros':
			self.wts = arr([np.zeros((sizes[i + 1],sizes[i] + 1)) for i in range(len(sizes) - 1)], dtype=object)
		elif init == 'random':
			self.wts = arr([.0025 * np.random.randn(sizes[i+1],sizes[i]+1) for i in range(len(sizes) - 1)], dtype=object)
		else:
			raise ValueError('NNetClassify.init_weights: ' + str(init) + ' is not a valid option for init')


## INSPECTORS ##################################################################


	def get_layers(self):
		S = arr([mat(self.wts[i]).shape[1] - 1 for i in range(len(self.wts))])
		S = concat((S, [mat(self.wts[-1]).shape[0]]))
		return S


## HELPERS #####################################################################


	def __responses(self, wts, X_in, sig, sig_0):
		"""
		Helper function that gets linear sum from previous layer (A) and
		saturated activation responses (Z) for a data point. Used in:
			train
		"""
		L = len(wts)
		constant_feat = np.ones((mat(X_in).shape[0],1)).flatten()	# constant feature
		# compute linear combination of inputs
		A = [arr([1])]
		Z = [concat((constant_feat, X_in))]

		for l in range(1, L):
			A.append(Z[l - 1].dot(wts[l - 1].T))					# compute linear combination of previous layer
			# pass through activation function and add constant feature
			Z.append(cols((np.ones((mat(A[l]).shape[0],1)),sig(A[l]))))

		A.append(arr(mat(Z[L - 1]) * mat(wts[L - 1]).T))
		Z.append(arr(sig_0(A[L])))									# output layer (saturate for classifier, not regressor)

		return A,Z


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

## RANDOM TESTING ##############################################################

	X,Y = load_data_from_csv('../data/gauss.csv', 4, float)
	X,Y = bootstrap_data(X, Y, 100000)
	# X,mu,scale = rescale(X)
	Xtr,Xte,Ytr,Yte = split_data(X, Y, .8)
	
	nc = NNetClassify(Xtr, Ytr, [4,5,5,5,5,5,5,5,4], init='random', max_steps=5000, activation='htangent')
	print(nc.get_weights())
	print(nc)
	print(nc.predict(Xte))
	print(nc.predict_soft(Xte))
	print(nc.err(Xte, Yte))

## DETERMINISTIC TESTING #######################################################

#	data = [[float(val) for val in row[:-1]] for row in csv.reader(open('../data/classifier-data.csv'))]
#	trd = np.asarray(data[0:40] + data[50:90] + data[100:140])
#	ted = np.asarray(data[40:50] + data[90:100] + data[140:150])
#	classes = [float(row[-1].lower()) for row in csv.reader(open('../data/classifier-data.csv'))]
#	trc = np.asarray(classes[0:40] + classes[50:90] + classes[100:140])
#	tec = np.asarray(classes[40:50] + classes[90:100] + classes[140:150])
#
#	trd,mu,scale = rescale(trd)
#	ted,mu,scale = rescale(ted)
#
#	print('nc')
#	nc = NNetClassify(trd, trc, [4,5,5,5,3], init='random', max_steps=5000, activation='htangent')
#	print(nc.get_weights())
#	print(nc)
#	print(nc.predict(ted))
#	print(nc.predict_soft(ted))
#	print(nc.err(ted, tec))


################################################################################
################################################################################
################################################################################
