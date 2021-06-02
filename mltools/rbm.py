import numpy as np
import copy

from .base import classifier
from .base import regressor
from .utils import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat

from scipy.special import expit
import matplotlib.pyplot as plt

################################################################################
## BASIC RBM    ################################################################
################################################################################

def _add1(X):
    return np.hstack( (np.ones((X.shape[0],1)),X) )

def _sigma(z): 
    return expit(z);
    #return 1.0/(1.0+np.exp(-z))


class crbm(object):
    """A restricted Boltzmann machine

    Attributes:
  
    """

    def __init__(self, nV,nH,nX, Wvh=None,bh=None,bv=None, Wvx=None,Whx=None):
        """Constructor for a restricted Boltzmann machine

        Parameters: 

        Properties:
        """
        if Wvh is None: Wvh = np.random.rand(nV,nH) * .001
        if Wvx is None: Wvx = np.random.rand(nV,nX) * .001
        if Whx is None: Whx = np.random.rand(nH,nX) * .001
        if bh  is None: bh  = np.zeros((nH,))
        if bv  is None: bv  = np.zeros((nV,))

        self.Wvh = Wvh
        self.Wvx = Wvx
        self.Whx = Whx
        self.bv = bv
        self.bh = bh



    def __repr__(self):
        to_return = 'Restricted Boltzmann machine, VxH={}x{}'.format(self.Wvh.shape[0],self.Wvh.shape[1])
        return to_return


    def __str__(self):
        to_return = 'Restricted Boltzmann machine, VxH={}x{}'.format(self.W.shape[0],self.W.shape[1])
        return to_return

    def nLayers(self):
        return 1

    @property
    def layers(self):
        """Return list of layer sizes, [N,H1,H2,...,C]

        N = # of input features
        Hi = # of hidden nodes in layer i
        C = # of output nodes (usually # of classes or 1)
        """
        if len(self.wts):
            layers = [self.W.shape[0], self.W.shape[1]]
            #layers = [self.wts[l].shape[1] for l in range(len(self.wts))]
            #layers.append( self.wts[-1].shape[0] )
        else:
            layers = []
        return layers

    @layers.setter
    def layers(self, layers):
        raise NotImplementedError
    # adapt / change size of weight matrices (?)



## CORE METHODS ################################################################
    # todo:  CD, BP; persistent CD?  make BP persistent?  others?
    #     :  estimate marginal likelihood in various ways?

    def marginals():
        raise NotImplementedError

    def marg_h(self, v, bh=None):
        if bh is None: bh = self.bh
        th = _sigma( v.dot(self.Wvh) + bh )  ## !!! regular rbm vs crbm?
        return th

    #@profile
    def marg_bp(self, maxiter=100, bv=None,bh=None,stoptol=1e-6):
        '''Estimate the singleton & pairwise marginals using belief propagation'''
        Wvh = self.Wvh        # pass in bv, bh to enable Whx etc?
        if bv is None: bv = self.bv
        if bh is None: bh = self.bh
        Mvh = np.empty(Wvh.shape); Mvh.fill(0.5);
        Mhv = Mvh.T.copy()
        tv, th = _sigma(self.bv), _sigma(self.bh)
        tvOld = 0*tv;
        for t in range(maxiter):
          # h to v:
          Lvh1 = (1 - Mhv).T * th   #Lvh1 = (1 - Mhv).T.dot( np.diag(th) )
          Lvh2 = Mhv.T * ( 1-th )   #Lvh2 = Mhv.T.dot( np.diag( 1-th ) )
          Mvh = _sigma( np.log( (np.exp(Wvh)*Lvh1 + Lvh2)/(Lvh1+Lvh2) ) )
          tv  = _sigma( bv + np.log( Mvh/(1-Mvh) ).sum(1) )
          if np.max(np.abs(tv-tvOld)) < stoptol: break;
          # v to h:
          Lhv1 = (1 - Mvh).T * tv  #Lhv1 = (1 - Mvh).T.dot( np.diag(tv) )
          Lhv2 = Mvh.T * (1-tv)    #Lhv2 = Mvh.T.dot( np.diag(1-tv) )
          Mhv  = _sigma( np.log( (np.exp(Wvh.T)*Lhv1+Lhv2)/(Lhv1+Lhv2) ) )
          th   = _sigma( bh + np.log( Mhv/(1-Mhv) ).sum(1) )
        Gsum = np.outer( 1-tv, 1-th ) * Mvh * Mhv.T
        Gsum+= np.outer( tv, 1-th)*(1-Mvh)*Mhv.T
        Gsum+= np.outer(1-tv,th)*Mvh*(1-Mhv.T)
        G    = np.exp(Wvh)*np.outer(tv,th)*(1-Mvh)*(1-Mhv.T)
        G   /= (Gsum+G)
        return G,tv,th


    def marg_cd(self, nstep=1,vinit=None, bv=None,bh=None, nchains=1):
        '''Estimate the singleton & pairwise marginals using gibbs sampling (for contrastive divergence)'''
        Wvh = self.Wvh        # pass in bv, bh to enable Whx etc?
        if bv is None: bv = self.bv
        if bh is None: bh = self.bh
        if vinit is None: raise NotImplementedError;  # todo: init using p(v)
        G,tv,th = 0,0,0
        for c in range(nchains):
          v = vinit;
          for s in range(nstep):
            ph = 1 / (1+np.exp(-v.dot(Wvh)-bh));
            h  = (np.random.rand(*ph.shape) < ph);
            pv = 1 / (1+np.exp(-Wvh.dot(h)-bv));
            v  = (np.random.rand(*pv.shape) < pv);
          tv += v; th += h; G += np.outer(v,h);
        return G,tv,th
        # TODO: variants: use p(h|v), or use all K samples


    def nll_gap(self, Xtr,Ytr, Xva,Yva):
        fe = np.mean( np.sum(Ytr*(self.bv + Xtr.dot(self.Wvx.T)),1) + 
               np.sum(np.log(1.0+np.exp( Ytr.dot(self.Wvh) + Xtr.dot(self.Whx.T) + self.bh ) ),1) )
        fe-= np.mean( np.sum(Yva*(self.bv + Xva.dot(self.Wvx.T)),1) + 
               np.sum(np.log(1.0+np.exp( Yva.dot(self.Wvh) + Xva.dot(self.Whx.T) + self.bh ) ),1) )
        return fe
       
    def err(self,X,Y):
        Y    = arr( Y )
        Yhat = arr( self.predict(X) )
        return np.mean(Yhat.reshape(Y.shape) != Y)

    def nll(self,X,Y):
        # TODO: fix; evaluate/estimate actual NLL?
        P = self.predictSoft(X);
        J = -np.mean( Y*np.log(P) + (1-Y)*np.log(1-P) );
        return J

    def predict(self, X):
        # Hard prediction.  TODO: create sampling function, MAP prediction function
        return self.predictSoft(X) > 0.5;

    def predictSoft(self, X):
        """Make 'soft' (per-class confidence) predictions of the rbm on data X.

        Args:
          X : MxN numpy array containing M data points with N features each

        Returns:
          P : MxC numpy array of C class probabilities for each of the M data
        """
        Y = np.zeros((X.shape[0],self.Wvx.shape[0]));
        for j in range(X.shape[0]):
            bxh = self.bh + self.Whx.dot(X[j,:].T)
            bxv = self.bv + self.Wvx.dot(X[j,:].T)
            mu = self.marg_h(Y[j,:],bxh)
            G,tv,th = self.marg_bp(5, bxv, bxh)
            Y[j,:] = tv;
        return Y



    def train(self, X, Y, Xv=None,Yv=None, stepsize=.01, stopGap=0.1, stopEpoch=100):
        """Train the (c)RBM

        Args:
          X : MxNx numpy array containing M data points with N features each
          Y : MxNv numpy array of targets (visible units) for each data point in X
          stepsize : scalar
              The stepsize for gradient descent (decreases as 1 / iter).
          stopTol : scalar 
              Tolerance for stopping criterion.
          stopIter : int 
              The maximum number of steps before stopping. 
          activation : str 
              'logistic', 'htangent', or 'custom'. Sets the activation functions.
        
        """
        # TODO: Shape & argument checking

        # outer loop of (mini-batch) stochastic gradient descent
        it, j = 1, 0                                # iteration number & data index
        nextPrint = 1                               # next time to print info
        done = 0                                    # end of loop flag
        nBatch = 40

        while not done:
            step_i = 3.0*stepsize / (2.0+it)        # step size evolution; classic 1/t decrease
           
            dWvh, dWvx, dWhx, dbv, dbh = 0.0, 0.0, 0.0, 0.0, 0.0 
            # stochastic gradient update (one pass)
            for jj in range(nBatch):
                #print('j={}; jj={};'.format(j,jj));
                j += 1
                if j >= Y.shape[0]: j=0; it+=1;
                # compute conditional model & required probabilities
                bxh = self.bh + self.Whx.dot(X[j,:].T)
                bxv = self.bv + self.Wvx.dot(X[j,:].T)
                mu = self.marg_h(Y[j,:],bxh)
                G,tv,th = self.marg_cd( 1, Y[j,:], bxv, bxh, 1)
                #G,tv,th = self.marg_bp( min(4+it,50), bxv, bxh )
                if (jj==1): #(np.random.rand() < .1):
                    plt.figure(1); 
                    plt.subplot(221); plt.imshow(X[j,:].reshape(28,28)); plt.title('Observed X'); plt.draw(); 
                    plt.subplot(222); plt.imshow(tv.reshape(28,28)); plt.title('Model Prob'); plt.draw();
                    plt.subplot(223); plt.imshow(Y[j,:].reshape(28,28)); plt.title('Visible Y'); plt.draw(); 
                    plt.pause(.01);
                # take gradient step:
                dWvh += (np.outer(Y[j,:], mu) - G)
                dWvx += (np.outer(Y[j,:], X[j,:]) - np.outer(tv,X[j,:]))
                dWhx += (np.outer(mu, X[j,:]) - np.outer(th,X[j,:]))
                dbv  += (Y[j,:] - tv)
                dbh  += (mu - th)

            self.Wvh += step_i * dWvh / nBatch
            self.Wvx += step_i * dWvx / nBatch
            self.Whx += step_i * dWhx / nBatch
            self.bv  += step_i * dbv / nBatch
            self.bh  += step_i * dbh / nBatch
            
            print('it {} : Gap = {}'.format(it,self.nll_gap(X,Y,Xv,Yv)));
            print('  {} {} {} {} {}'.format(np.mean(self.Wvx**2),np.mean(self.Whx**2),np.mean(self.Wvh**2),np.mean(self.bv**2),np.mean(self.bh**2)));

            Jtr,Jva = 0,0 #self.nll(X,Y),self.nll(Xv,Yv);
            if it >= nextPrint:
                print('it {} : Gap = {}'.format(it,self.nll_gap(X,Y,Xv,Yv)));
                print('  {} {} {} {} {}'.format(np.mean(self.Wvx**2),np.mean(self.Whx**2),np.mean(self.Wvh**2),np.mean(self.bv**2),np.mean(self.bh**2)));
                #print('it {} : Jtr = {} / Jva = {}'.format(it,Jtr,Jva))
                nextPrint += 1; #*= 2

            # check if finished
            done = (it > 1) and ((Jva - Jtr) > stopGap) or it >= stopEpoch
            #it += 1   # counting epochs elsewhere now




    #def err_k(self, X, Y):
    #    """Compute misclassification error rate. Assumes Y in 1-of-k form.  """
    #    return self.err(X, from1ofK(Y,self.classes).ravel())
    #    
    #    
    #def mse(self, X, Y):
    #    """Compute mean squared error of predictor 'obj' on test data (X,Y).  """
    #    return mse_k(X, to1ofK(Y))
    #
    #
    #def mse_k(self, X, Y):
    #    """Compute mean squared error of predictor; assumes Y is in 1-of-k format.  """
    #    return np.power(Y - self.predictSoft(X), 2).sum(1).mean(0)


## MUTATORS ####################################################################



################################################################################
################################################################################
################################################################################

