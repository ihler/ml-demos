import numpy as np

from .base import classifier
from .base import regressor
from .utils import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat
from numpy import ceil


################################################################################
## DECISION TREES ##############################################################
################################################################################

## TODO: data weights




class treeBase(object):

    def __init__(self, *args, **kwargs):
        """Constructor for decision tree base class

        Args:
          *args, **kwargs (optional): passed to train function

        Properties (internal use only)
           L,R (arr): indices of left & right child nodes in the tree
           F,T (arr): feature index & threshold for decision (left/right) at this node
             P (arr): for leaf nodes, P[n] holds the prediction for leaf node n
        """
        self.L = arr([])           # indices of left children
        self.R = arr([])           # indices of right children
        self.F = arr([])           # feature to split on (-1 = leaf = predict)
        self.T = arr([])           # threshold to split on 
        self.P = arr([])           # prediction value for node
        self.sz = 0                 # size; also next node during construction
   
        if len(args) or len(kwargs):     # if we were given optional arguments,
            self.train(*args, **kwargs)    #  just pass them through to "train"
 
    
    def __repr__(self):
        to_return = 'Decision Tree\n'
        if len(self.T) > 8: return self.str_short()
        else:               return self.str_long()

    def __printTree(self,node,indent):
        to_return = ''
        if (self.F[node] == -1):
            to_return += indent+'Predict {}\n'.format(self.P[node])
        else:
            to_return += indent+'if x[{:d}] < {:f}:\n'.format(int(self.F[node]),self.T[node])
            to_return += self.__printTree(self.L[node],indent+'  ')
            to_return += indent+'else:\n'
            to_return += self.__printTree(self.R[node],indent+'  ')
        return to_return

    __str__ = __repr__

    def str_short(self): 
        ''' "Short" string representation of the decision tree (thresholds only)'''
        return 'Thresholds: {}'.format(
                '[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]' 
                .format(self.T[0], self.T[1], self.T[-1], self.T[-2]));

    def str_long(self): 
        ''' "Long" string representation of the decision tree (if-then-else)'''
        return self.__printTree(0,'  ');

## CORE METHODS ################################################################


    def train(self, X, Y, minParent=2, maxDepth=np.inf, minLeaf=1, nFeatures=None):
        """ Train a decision-tree model

        Args:
          X (arr) : M,N numpy array of M data points with N features each
          Y (arr) : M, or M,1 array of target values for each data point
          minParent (int): Minimum number of data required to split a node. 
          minLeaf   (int): Minimum number of data required to form a node
          maxDepth  (int): Maximum depth of the decision tree. 
          nFeatures (int): Number of available features for splitting at each node.
        """
        n,d = mat(X).shape
        nFeatures = min(nFeatures,d) if nFeatures else d

        sz = int(min(ceil(2.0*n/minLeaf), 2**(maxDepth + 1)))   # pre-allocate storage for tree:
        self.L, self.R, self.F, self.T = np.zeros((sz,),dtype=int), np.zeros((sz,),dtype=int), np.zeros((sz,),dtype=int), np.zeros((sz,))
        sh = list(Y.shape)
        sh[0] = sz
        self.P = np.zeros(sh,dtype=Y.dtype) #np.zeros((sz,1))  # shape like Y 
        self.sz = 0              # start building at the root

        self.__train_recursive(X, Y, 0, minParent, maxDepth, minLeaf, nFeatures)

        self.L = self.L[0:self.sz]                              # store returned data into object
        self.R = self.R[0:self.sz]                              
        self.F = self.F[0:self.sz]
        self.T = self.T[0:self.sz]
        self.P = self.P[0:self.sz]



    def predict(self, X):
        """Make predictions on the data in X

        Args:
          X (arr): MxN numpy array containing M data points of N features each

        Returns:
          arr : M, or M,1 vector of target predictions
        """
        return self.__predict_recursive(X, 0)


    
## HELPERS #####################################################################


    #TODO: compare for numerical tolerance
    def __train_recursive(self, X, Y, depth, minParent, maxDepth, minLeaf, nFeatures):
        """ Recursive helper method that recusively trains the decision tree. """
        n,d = mat(X).shape

        # check leaf conditions...
        if n < max(minParent,2*minLeaf) or depth >= maxDepth or np.var(Y-Y[0])==0: return self.__build_leaf(Y)

        best_val = np.inf
        best_feat = -1
        try_feat = np.random.permutation(d)

        # ...otherwise, search over (allowed) features
        for i_feat in try_feat[0:nFeatures]:
            dsorted = arr(np.sort(X[:,i_feat].T)).ravel()                # sort data...
            pi = np.argsort(X[:,i_feat].T)                               # ...get sorted indices...
            tsorted = Y[pi]                                              # ...and sort targets by feature ID
            can_split = np.append(arr(dsorted[:-1] != dsorted[1:]), 0)   # which indices are valid split points?
            # TODO: numeric comparison instead?
            can_split[np.arange(0,minLeaf-1)] = 0
            can_split[np.arange(n-minLeaf,n)] = 0   # TODO: check

            if not np.any(can_split):          # no way to split on this feature?
                continue

            # find min weighted variance among split points
            val,idx = self.data_impurity(tsorted, can_split)

            # save best feature and split point found so far
            if val < best_val:
                best_val, best_feat, best_thresh = val, i_feat, (dsorted[idx] + dsorted[idx + 1]) / 2.0

        # if no split possible, output leaf (prediction) node
        if best_feat == -1: return self.__build_leaf(Y)

        # split data on feature i_feat, value (tsorted[idx] + tsorted[idx + 1]) / 2
        self.F[self.sz] = best_feat
        self.T[self.sz] = best_thresh
        go_left = X[:,self.F[self.sz]] < self.T[self.sz]  # index data going left & right
        go_right= np.logical_not(go_left)
        my_idx = self.sz      # save current node index for left,right pointers
        self.sz += 1          # advance to next node to build subtree

        # recur left
        self.L[my_idx] = self.sz    
        self.__train_recursive(X[go_left,:], Y[go_left], depth+1, minParent, maxDepth, minLeaf, nFeatures)

        # recur right
        self.R[my_idx] = self.sz    
        self.__train_recursive(X[go_right,:], Y[go_right], depth+1, minParent, maxDepth, minLeaf, nFeatures)

        return


    def __predict_recursive(self, X, pos):
        """Recursive helper function for finding leaf nodes during prediction """
        m,n = X.shape
        sh = list(self.P.shape)
        sh[0] = m
        Yhat = np.zeros(sh,dtype=self.P.dtype)

        if self.F[pos] == -1:        # feature to compare = -1 => leaf node
            Yhat[:] = self.P[pos]    # predict stored value
        else:
            go_left = X[:,self.F[pos]] < self.T[pos]  # which data should follow left split?
            Yhat[go_left]  = self.__predict_recursive(X[go_left,:],  self.L[pos])
            go_right = np.logical_not(go_left)        # other data go right:
            Yhat[go_right] = self.__predict_recursive(X[go_right,:], self.R[pos])

        return Yhat


    def __build_leaf(self, Y):
        """Helper function for setting parameters at leaf nodes during train"""
        self.F[self.sz] = -1
        self.P[self.sz] = self.data_average(Y)      # TODO: convert to predict f'n call
        self.sz += 1




################################################################################
# REGRESSION SUBCLASS ##########################################################
################################################################################

class treeRegress(treeBase,regressor):

    @staticmethod
    def weighted_avg(Y):
        return np.mean(Y, axis=0)

    @staticmethod
    def min_weighted_var(tsorted, can_split):
        """(weighted) variance impurity score function for regression (mse)
           returns (value,index) of the split with the lowest weighted variance
        """
        # compute mean up to and past position j (for j = 0..n)
        n = tsorted.shape[0]
        y_cum_to = np.cumsum(tsorted, axis=0)
        y_cum_pa = y_cum_to[-1] - y_cum_to
        count_to = np.arange(1.0,n+1)
        count_pa = np.arange(1.0*n - 1, -1, -1)
        count_pa[-1] = 1.0
        if len(y_cum_to.shape)>1:
            count_to, count_pa = count_to.reshape(-1,1), count_pa.reshape(-1,1)
        mean_to = y_cum_to / count_to; 
        mean_pa = y_cum_pa / count_pa; 

        # compute variance up to, and past position j (for j = 0..n)
        y2_cum_to = np.cumsum(np.power(tsorted, 2), axis=0)
        y2_cum_pa = y2_cum_to[-1] - y2_cum_to
        var_to = (y2_cum_to - 2 * mean_to * y_cum_to + count_to * np.power(mean_to, 2)) / count_to
        var_pa = (y2_cum_pa - 2 * mean_pa * y_cum_pa + count_pa * np.power(mean_pa, 2)) / count_pa
        if len(var_to.shape)>1:
            var_to = var_to.sum(1,keepdims=True)   # take total variance across dimensions if
            var_pa = var_pa.sum(1,keepdims=True)   #  multivariate targets

        # find minimum weighted variance among all split points
        weighted_variance = count_to/n * var_to + count_pa/n * var_pa
        weighted_variance[-1] = np.inf
        weighted_variance[can_split==0] = np.inf   # find only splittable points
        idx = np.nanargmin(weighted_variance)      # use nan version to ignore any nan values
        val = float(weighted_variance[idx])

        return (val,idx)

    def __init__(self, *args,**kwargs):
      """Decision tree for regression

      See train for arguments
      """
      treeBase.__init__(self,*args,**kwargs)
 
    train = treeBase.train
    predict = treeBase.predict

    data_impurity = min_weighted_var
    data_average  = weighted_avg


################################################################################
# CLASSIFICATION SUBCLASS ######################################################
################################################################################

class treeClassify(treeBase,classifier):
    def __init__(self, *args, **kwargs):
        """Constructor for decision tree regressor; all args passed to train"""
        self.classes = []
        treeBase.__init__(self,*args,**kwargs);
        #super(treeClassify,self).__init__(*args,**kwargs);

    def train(self, X, Y, *args,**kwargs):
        """ Train a decision-tree model

        Parameters
        ----------
        X : M x N numpy array of M data points with N features each
        Y : numpy array of shape (M,) that contains the target values for each data point
        minParent : (int)   Minimum number of data required to split a node. 
        minLeaf   : (int)   Minimum number of data required to form a node
        maxDepth  : (int)   Maximum depth of the decision tree. 
        nFeatures : (int)   Number of available features for splitting at each node.
        """
        self.classes = list(np.unique(Y)) if len(self.classes) == 0 else self.classes
        treeBase.train(self,X,to1ofK(Y,self.classes).astype(float),*args,**kwargs);

    def predict(self,X):
        """Make predictions on the data in X

        Args:
          X (arr): MxN numpy array containing M data points of N features each

        Returns:
          arr : M, or M,1 vector of target predictions
        """
        return classifier.predict(self,X)

    def predictSoft(self,X):
        """Make soft predictions on the data in X

        Args:
          X (arr): MxN numpy array containing M data points of N features each

        Returns:
          arr : M,C array of C class probabiities for each data point
        """
        return treeBase.predict(self,X);

    @staticmethod
    def entropy(tsorted, can_split):
        """Return the value and index of the minimum of the Shannon entropy impurity score"""
        n = tsorted.shape[0]
        eps = np.spacing(1)
        #y_left = np.cumsum(to1ofK(tsorted, self.classes), axis=0).astype(float)
        y_left = np.cumsum(tsorted, axis=0)
        y_right = y_left[-1,:] - y_left         # construct p(class) for each possible split
        wts_left = np.arange(1.0,n+1)     # by counting & then normalizing by left/right sizes
        y_left /= wts_left.reshape(-1,1)
        tmp = n - wts_left
        tmp[-1] = 1
        y_right /= tmp.reshape(-1,1)
        wts_left /= n

        h_root  = -np.dot(y_left[-1,:], np.log(y_left[-1,:] + eps).T)
        h_left  = -np.sum(y_left * np.log(y_left + eps), axis=1)
        h_right = -np.sum(y_right * np.log(y_right + eps), axis=1)

        IG = h_root - (wts_left * h_left + (1.0-wts_left) * h_right)
        val = np.max((IG + eps) * can_split)
        idx = np.argmax((IG + eps) * can_split)
        return (h_root-val,idx)

    @staticmethod
    def weighted_avg(Y, reg=0.5):
        """Return the weighted average probability vector of the classes"""
        p = np.sum(Y,axis=0) + reg
        return p / p.sum()


    data_impurity = entropy
    data_average  = weighted_avg



################################################################################
################################################################################
################################################################################

def plotTree2D(tree,bbox=None,styles=['k-'],colors=None,alpha=.3):
    '''Plot a decision tree on two features.
       Args:
         bbox = [xmin,xmax,ymin,ymax] : list bounds of full plot domain
         styles : list of string styles for decision boundary lines; ['none'] to skip
         colors : list of colors for leaf regions; ['none']=skip, None=automatic
         alpha  : opacity for leaf color regions; default 0.3
    '''
    import matplotlib.pyplot as plt
    if isinstance(styles,str): styles=[styles];
    if bbox is None: bbox = list(plt.gca().axis());
    if colors is None:
        try:   # for classifiers, get colors as in plotClassify2D
            classes = np.array(tree.classes);
            vmin,vmax = classes.min(),classes.max()
            cmap = plt.cm.get_cmap()
            cvals = (classes - vmin)/(vmax-vmin+1e-100)
            colors = [cmap(c) for c in cvals];
        except Exception: pass
        
    def __recursePlot(tree,depth,node,bbox):
        f = tree.F[node];
        if f < 0:  # at leaf nodes, plot a colored rectangle; colors=['none'] to not draw
            color = tree.P[node] if colors==None else colors[np.argmax(tree.P[node]) % len(colors)];
            r=plt.Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2], alpha=alpha,facecolor=color, edgecolor='none');
            plt.gca().add_patch(r)
        else:      # at internal nodes, draw dividing lines; styles=['none'] to not draw
            if f==0:
                plt.plot([tree.T[node],tree.T[node]],[bbox[2],bbox[3]],styles[depth%len(styles)])
            elif f==1:
                plt.plot([bbox[0],bbox[1]],[tree.T[node],tree.T[node]],styles[depth%len(styles)])
            bboxL,bboxR = bbox[:],bbox[:]; bboxR[2*f]=tree.T[node]; bboxL[2*f+1]=tree.T[node];
            __recursePlot(tree,depth+1,tree.L[node],bboxL);
            __recursePlot(tree,depth+1,tree.R[node],bboxR);
    __recursePlot(tree,0,0,bbox);


