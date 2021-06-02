import numpy as np

from .base import classifier
from .base import regressor
from .utils import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat


################################################################################
## KERNEL CLASSIFY #############################################################
################################################################################

def gaussian(A,B,sig):
    ma,na = (1,A.shape[0]) if len(A.shape)==1 else (A.shape[0],A.shape[1])
    mb,nb = (1,B.shape[0]) if len(B.shape)==1 else (B.shape[0],B.shape[1])
    return np.exp( -np.sum( ((A.reshape(ma,na,1)-B.T.reshape(1,nb,mb))/sig)**2 , axis=1) );
def linear(A,B):
    return np.dot(A,B.T);
def poly(A,B,d):
    return (1+np.dot(A,B.T))**d;


class kernelClassify2(classifier):   # BINARY Kernel SVM


    def __init__(self, *args, **kwargs):
        """
        Constructor for kernelClassify object.  

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           kernel  : function handle for computing kernel similiarity
           X       : mxn numpy array training data matrix
           alpha   : lagrange multipliers of the SVM
        """
        self.classes = []
        self.kernel = linear; #lambda a,b : np.dot(a,b);
        self.X      = np.array([])
        self.alpha = np.array([])

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


    def __repr__(self):
        str_rep = 'kernelClassify model, {} data, {} features\n{}'.format(
                   self.X.shape(0), self.X.shape(1), self.theta)
        return str_rep


    def __str__(self):
        str_rep = 'kernelClassify model, {} data, {} features\n{}'.format(
                   self.X.shape(0), self.X.shape(1), self.theta)
        return str_rep

    def predictSoft(self, X):
        """ Soft prediction (response) of the SVM """
        m = X.shape[0]
        r = np.zeros(m)
        for j in range(m): r[j] = np.dot( self.alpha*self.Y,self.kernel(self.X,X[j]) ) + self.bias
        return np.vstack((-r,r)).T


    def train(self,X,Y, R=1.0, tol=1e-6, maxIter=1000):
        """SMO algorithm for optimizing the dual form of a kernel SVM
          X,Y (nparrays): data features and targets (+/- 1)
          R   (1.0) : bound on Lagrange multipliers, or scale coefficient on the slack variables
          tol (1e-6): convergence tolerance
          maxIter (1000): maximum number of iterations before exiting
        """
        self.alpha = 0*Y; self.bias = 0;
        self.classes = np.unique(Y);
        assert( len(self.classes) == 2 ) # Doesn't work on non-binary classification yet
        self.X = X; self.Y = 0*Y - (Y==self.classes[0]) + (Y==self.classes[1]); 
        
        for it in range (maxIter):
            nUpdated = 0
            X,Y = self.X, self.Y
            m = Y.shape[0]
            for i in np.random.permutation(m):                  # pick a data point that violates the constraints:
                Ei = self.predictSoft( X[i:i+1] )[:,1] - Y[i];
                if ( (Ei*Y[i]<-tol and self.alpha[i]<R) or (Ei*Y[i] > tol and self.alpha[i]>0) ):
                    for j in np.random.permutation(m):          #   and another to update jointly with it
                        if j == i:  continue   
                        Ej = self.predictSoft( X[j:j+1] )[:,1] - Y[j];
                        ai, aj = self.alpha[i], self.alpha[j]   # find the constraint bounds for updating alpha[j]
                        if Y[i] != Y[j]: L,H = max(0,self.alpha[j]-self.alpha[i]),min(R,R+self.alpha[j]-self.alpha[i])
                        else:            L,H = max(0,self.alpha[i]+self.alpha[j]-R),min(R,self.alpha[i]+self.alpha[j])
                        if L>=H: continue      # if bounds are empty, we won't update
                        Xii, Xij, Xjj = self.kernel(X[i],X[i]), self.kernel(X[i],X[j]), self.kernel(X[j],X[j]);
                        eta = 2*Xij - Xii - Xjj
                        if eta >= 0: continue;
                        self.alpha[j] -= Y[j]*(Ei-Ej)/eta;
                        self.alpha[j] = max(min(self.alpha[j],H),L)    # enforce alpha[j] in range [L,H]
                        if (self.alpha[j]-aj)**2 < tol: continue       # if alpha[j] doesn't change, we can just quit
                        self.alpha[i] += Y[i]*Y[j]*(aj-self.alpha[j])  # ow, update alpha[i] to match
                        bi = self.bias - Ei - Y[i]*(self.alpha[i]-ai)*Xii - Y[j]*(self.alpha[j]-aj)*Xij;
                        bj = self.bias - Ej - Y[i]*(self.alpha[i]-ai)*Xij - Y[j]*(self.alpha[j]-aj)*Xjj;
                        if 0<self.alpha[i] and self.alpha[i] < R: self.bias=bi;
                        elif 0<self.alpha[j] and self.alpha[j] < R: self.bias=bj;
                        else: self.bias = (bi+bj)/2.
                        Ei = self.predictSoft( X[i:i+1] )[:,1] - Y[i]; # Update Ei after changes
                        nUpdated += 1;
            # discard non-support vectors after each iteration?  (still optimal?)
            #self.X, self.Y, self.alpha = self.X[self.alpha>0], self.Y[self.alpha>0], self.alpha[self.alpha>0]
            if nUpdated == 0: break;  # if we iterated and found no violations, we're done
        # We can also evaluate the bias after convergence using the current non-boundary support vectors:            
        #ins = (self.alpha>0)&(self.alpha<R)
        #self.bias = 0;
        #self.bias = -np.mean( (self.predictSoft(X)[:,1]-Y)[ins] )
        # Discard non-support vectors after convergence?
        self.X, self.Y, self.alpha = self.X[self.alpha>0], self.Y[self.alpha>0], self.alpha[self.alpha>0]

        if it==maxIter-1: print("Warning: convergence conditions not reached."); # TODO: should raise a warning about convergence failure 
        else: print("Done; "+str(it)+" iters");


