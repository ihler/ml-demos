import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod

from .utils import from1ofK


################################################################################
## KMEANS ######################################################################
################################################################################


def kmeans(X, K, init='random', max_iter=100):
    """
    Perform K-means clustering on data X.

    Parameters
    ----------
    X : numpy array
        N x M array containing data to be clustered.
    K : int
        Number of clusters.
    init : str or array (optional)
        Either a K x N numpy array containing initial clusters, or
        one of the following strings that specifies a cluster init
        method: 'random' (K random data points (uniformly) as clusters),
        'farthest' (choose cluster 1 uniformly, then the point farthest
        from all cluster so far, etc.), or 'k++' (choose cluster 1 
        uniformly, then points randomly proportional to distance from
        current clusters).
    max_iter : int (optional)
        Maximum number of optimization iterations.

    Returns (as tuple)
    -------
    z    : N x 1 array containing cluster numbers of data at indices in X.
    c    : K x M array of cluster centers.
    sumd : (scalar) sum of squared euclidean distances.
    """
    n,d = twod(X).shape

    # First, initialize the clusters to something:
    if type(init) is str:
        init = init.lower()
        if init == 'random':
            pi = np.random.permutation(n)
            c = X[pi[0:K],:]
        elif init == 'farthest':
            c = k_init(X, K, True)
        elif init == 'k++':
            c = k_init(X, K, False)
        else:
            raise ValueError('kmeans: value for "init" ( ' + init +  ') is invalid')
    else:
        c = init

    # Now, optimize the objective using coordinate descent:
    iter = 1
    done = (iter > max_iter)
    sumd = np.inf
    sum_old = np.inf

    z = np.zeros((n,))
    #print c

    while not done:
        sumd = 0
        
        for i in range(n):
            # compute distances for each cluster center
            dists = np.sum( (c - twod(X[i,:]))**2 , axis=1)
            #dists = np.sum(np.power((c - np.tile(X[i,:], (K,1))), 2), axis=1)
            val = np.min(dists, axis=0)                         # assign datum i to nearest cluster
            z[i] = np.argmin(dists, axis=0)
            sumd = sumd + val

        #print z
        for j in range(K):                              # now update each cluster center j...
            if np.any(z == j):
                c[j,:] = np.mean(X[(z == j).flatten(),:], 0)# ...to be the mean of the assigned data...
            else:
                c[j,:] = X[int(np.floor(np.random.rand())),:]   # ...or random restart if no assigned data

        done = (iter > max_iter) or (sumd == sum_old)
        sum_old = sumd
        iter += 1

    return z, c, sumd
            

def k_init(X, K, determ=False, distance=None):
    """
    Distance based initialization. Randomly choose a start point, then:
    if determ == True: choose point farthest from the clusters chosen so
    far, otherwise: randomly choose new points proportionally to their
    distance.

    Parameters
    ----------
    X : (m,n) numpy array of m data points with n features
    K : int, number of clusters to select
    determ : bool; False = sample based on distance
    distance: distance function distance(X,v) (default: Euclidean)

    Returns
    -------
    c : numpy array
        K x M array of cluster centers.
    """
    m,n = twod(X).shape
    clusters = np.zeros((K,n))
    clusters[0,:] = X[int(np.floor(np.random.rand() * m)),:]            # take random point as first cluster
    if distance is None:
        distance = lambda X,v: np.sum(np.power(X - v[np.newaxis,:],2),axis=1).ravel()
    dist = distance(X,clusters[0,:]); #np.sum(np.power((X - np.ones((m,1)) * clusters[0,:]), 2), axis=1).ravel()
    #print 'dist:',dist

    for i in range(1,K):
        #print dist
        #print np.argmax(dist)
        if determ:
            j = np.argmax(dist)                                 # choose farthest point...
        else:
            pr = np.cumsum(np.array(dist));                             # ...or choose a random point by distance
            pr = pr / pr[-1]
            j = np.where(np.random.rand() < pr)[0][0]

        clusters[i,:] = X[j,:]                                  # update that cluster
        # update min distances
        #new_dist = np.sum(np.power((X - np.ones((m,1)) * clusters[i,:]), 2), axis=1).ravel()
        dist = np.minimum(dist, distance(X,clusters[i,:])) #new_dist)
        #print "dist",dist

    return clusters



################################################################################
## AGGLOMERATIVE ###############################################################
################################################################################


def agglomerative(X, K, method='means', join=None, dist=None):
    """
    Perform hierarchical agglomerative clustering.

    Parameters
    ----------
    X : numpy array
        N x M array of data to be clustered.
    K : int
        The number of clusters into which data should be grouped.
    method : str (optional)
        str that specifies the method to use for calculating distance between
        clusters. Can be one of: 'min', 'max', 'means', or 'average'.
    join : numpy array (optional)
        N - 1 x 3 that contains a sequence of joining operations. Pass to avoid
        reclustering for new X.
        dist: function dist(a,b) returning the distance between data a & b (default: Euclidean)

    Returns (tuple)
    -------
    z    : N x 1 array of cluster assignments.
    join : N - 1 x 3 array that contains the sequence of joining operations 
        peformed by the clustering algorithm.
    """
    m,n = twod(X).shape                 # get data size
    D = np.zeros((m,m)) + np.inf        # store pairwise distances b/w clusters (D is an upper triangular matrix)
    z = arr(range(m))                   # assignments of data
    num = np.ones(m)                    # number of data in each cluster
    mu = arr(X.copy())                  # centroid of each cluster
    method = method.lower()

    if type(join) == type(None):        # if join not precomputed

        join = np.zeros((m - 1, 3))     # keep track of join sequence
        # use standard Euclidean distance
        if dist is None: dist = lambda a,b: np.sum(np.power(a - b, 2))
        for i in range(m):              # compute initial distances
            for j in range(i + 1, m):
                D[i][j] = dist(X[i,:], X[j,:])


        opn = np.ones(m)                # store list of clusters still in consideration
        val,k = np.min(D),np.argmin(D)  # find first join (closest cluster pair)
        
        for c in range(m - 1):
            i,j = np.unravel_index(k, D.shape)
            join[c,:] = arr([i, j, val])

            # centroid of new cluster
            mu_new = (num[i] * mu[i,:] + num[j] * mu[j,:]) / (num[i] + num[j])

            # compute new distances to cluster i
            for jj in np.where(opn)[0]:
                if jj in [i, j]:
                    continue

                # sort indices because D is an upper triangluar matrix
                idxi = tuple(sorted((i,jj)))    
                idxj = tuple(sorted((j,jj)))    
                    
                if method == 'min':
                    D[idxi] = min(D[idxi], D[idxj])     # single linkage (min dist)
                elif method == 'max':
                    D[idxi] = max(D[idxi], D[idxj])     # complete linkage (max dist)
                elif method == 'means':
                    D[idxi] = dist(mu_new, mu[jj,:])    # mean linkage (dist b/w centroids)
                elif method == 'average':
                    # average linkage
                    D[idxi] = (num[i] * D[idxi] + num[j] * D[idxj]) / (num[i] + num[j])

            opn[j] = 0                      # close cluster j (fold into i)
            num[i] = num[i] + num[j]        # update total membership in cluster i to include j
            mu[i,:] = mu_new                # update centroid list

            # remove cluster j from consideration as min
            for ii in range(m):
                if ii != j:
                    # sort indices because D is an upper triangular matrix
                    idx = tuple(sorted((ii,j))) 
                    D[idx] = np.inf

            val,k = np.min(D), np.argmin(D) # find next smallext pair

    # compute cluster assignments given sequence of joins
    for c in range(m - K):
        z[z == join[c,1]] = join[c,0]

    uniq = np.unique(z)
    for c in range(len(uniq)):
        z[z == uniq[c]] = c

    return z, join



def dendsort(join):
    """Return the dendrogram-sorted order from a full sequence of agglomerative joins"""
    def ssort(node, t):   # just recursively process the merge steps
        while t>=0 and join[t][0] != node: t -= 1;
        if t < 0: return [node],0,1;  # leaf condition
        else: 
            c0,s0,n0 = ssort(join[t][0],t-1);  # or, recurse both sides of join
            c1,s1,n1 = ssort(join[t][1],t-1);
            if n0==1 and n1==1: return c0+c1,join[t][2],2  # put them in some order: 
            if n1==1: return c0+c1,(s0*n0+join[t][2])/(n0+1),n0+1 # leaves go on right
            if n0==1: return c1+c0,(s1*n1+join[t][2])/(n1+1),n1+1 # & "tighter" joins go left
            if s0 < s1: return c0+c1,(s0*n0+s1*n1+join[t][2])/(n0+n1),n0+n1
            return c1+c0,(s0*n0+s1*n1+join[t][2])/(n0+n1),n0+n1
    return ssort(join[-1][0],len(join)-1)[0]


def dendrogram(join, ax=None,**kwargs):
    """Plot a dendrogram for a given join sequence (see agglomorate)"""
    import matplotlib.pyplot as plt
    order = np.array(dendsort(join),dtype=int); m = len(order);
    x = np.arange(m,dtype=float); x[order] = np.arange(m); y = np.zeros((m,));
    if ax is None: ax=plt.gca();
    for j in join:
        j0,j1 = int(j[0]),int(j[1]);
        x0,y0,x1,y1,h = x[j0],y[j0],x[j1],y[j1],j[2];
        ax.plot( [x0,x0,x1,x1],[y0,h,h,y1], 'b-' ,**kwargs);
        x[j0], y[j0] = .5*(x0+x1), h;
    ax.set_xticks(np.arange(m)); ax.set_xticklabels(order);



################################################################################
## EXPECTATION-MAXIMIZATION ####################################################
################################################################################


def gmmEM(X, K, init='random', max_iter=100, tol=1e-6, reg=1e-6):
    """
    Perform Gaussian mixture EM (expectation-maximization) clustering on data X.

    Parameters
    ----------
    X : numpy array
        N x M array containing data to be clustered.
    K : int
        Number of clusters.
    init : str, array, or dict (optional)
        * (k,n) numpy array containing initial cluster means, OR
        * dict containing cluster initializations (see output "T") OR
        * one of the following strings that specifies a cluster init:
          method: 'random' (K random data points (uniformly) as clusters)
                  'farthest' (choose cluster 1 uniformly, then the point farthest
                       from all cluster so far, etc.)
                  'k++' (choose cluster 1 uniformly, then points randomly 
                       proportional to distance from current clusters).
    max_iter : int (optional)
        Maximum number of iterations.
    tol : scalar (optional)
        Stopping tolerance.

    Returns
    -------
    z    : 1 x N numpy array of cluster assignments (int indices).
    T    : {'pi': np.array, 'mu': np.array, 'sig': np.array} : Gaussian component parameters
    soft : numpy array; soft assignment probabilities (rounded for assign)
    ll   : float; Log-likelihood under the returned model.
    """
    # TODO: fix: pass in initial *clusters* (pi,mu,sig); sig should be K x D x D, not DxDxK
    # init
    N,D = twod(X).shape                 # get data size

    sig = np.zeros((D,D,K));
    for c in range(K): sig[:,:,c] = np.eye(D)
    alpha = np.ones(K) / K

    if type(init) is str:
        init = init.lower()
        if init == 'random':
            pi = np.random.permutation(N)
            mu = 1.*X[pi[0:K],:]
        elif init == 'farthest':
            mu = k_init(X, K, True)
        elif init == 'k++':
            mu = k_init(X, K, False)
        else:
            raise ValueError('gmmEM: value for "init" ( ' + init +  ') is invalid')
    else:
        try:
            alpha,mu,sig = 1.*arr(init['pi']),1.*arr(init['mu']),1.*arr(init['sig']);
        except Exception:
            mu = (1.*arr(init))    # assumes a numpy array

    R = np.zeros((N,K))

    iter,ll,ll_old = 1, -np.inf, -np.inf
    done = iter > max_iter
    C = np.log(2 * np.pi) * D / 2

    while not done:
        ll = 0
        for c in range(K):
            # compute log prob of all data under model c
            V = X - np.tile(mu[c,:], (N,1))         
            R[:,c] = -0.5 * np.sum((V.dot(np.linalg.inv(sig[:,:,c]))) * V, axis=1) - 0.5 * np.log(np.linalg.det(sig[:,:,c])) + np.log(alpha[c]) - C

        # avoid numerical issues by removing constant 1st
        mx = R.max(1)
        R -= mx[:,np.newaxis]; #np.tile(twod(mx).T, (1,K))
        # exponentiate and compute sum over components
        R = np.exp(R)
        nm = R.sum(1)
        # update log-likelihood of data
        ll = np.sum(np.log(nm) + mx)
        R /= nm[:,np.newaxis]; #np.tile(twod(nm).T, (1,K))     # normalize to give membership probabilities

        alpha = R.sum(0)                    # total weight for each component
        for c in range(K):
            # weighted mean estimate
            mu[c,:] = (R[:,c] / alpha[c]).T.dot(X)
            tmp = X - np.tile(mu[c,:], (N,1))
            # weighted covar estimate
            sig[:,:,c] = tmp.T.dot(tmp * np.tile(twod(R[:,c]).T / alpha[c], (1,D))) + reg * np.eye(D)
        alpha /= N

        # stopping criteria
        done = (iter >= max_iter) or np.abs(ll - ll_old) < tol
        ll_old = ll
        iter += 1

    z = from1ofK(R)
    soft = R
    T = {'pi': alpha, 'mu': mu, 'sig': sig}
        
    return z, T, soft, ll


