import numpy as np
import matplotlib.pyplot as plt

from numpy import atleast_2d as twod


################################################################################
##  PLOTTING FUNCTIONS #########################################################
################################################################################


def plotClassify2D(learner, X, Y, pre=lambda x: x, ax=None, nGrid=128, cm=None, bgalpha=0.3, soft=False, **kwargs):
    """
    Plot data and classifier outputs on two-dimensional data.
    This function plots data (X,Y) and learner.predict(X, Y)
    together. The learner is is predicted on a dense grid
    covering the data X, to show its decision boundary.

    Parameters
    ----------
    learner : a classifier with "predict" function and optionally "classes" list
    X : (m,n) numpy array of data (m points in n=2 dimension)
    Y : (m,) or (m,1) int array of class values OR (m,c) array of class probabilities (see predictSoft)
    pre   : function object (optional) applied to X before learner.predict()
    ax    : a matplotlib axis / plottable object (optional)
    nGrid : density of 2D grid points (default 128)
    soft  : use predictSoft & blend colors (default: False => use predict() and show decision regions)
    bgalpha: alpha transparency (1=opaque, 0=transparent) for decision function image
    cm    : pyplot colormap (default: None = use default colormap)
    [other arguments will be passed through to the pyplot scatter function on the data points]
    """

    if twod(X).shape[1] != 2:
        raise ValueError('plotClassify2D: function can only be called using two-dimensional data (features)')
    # make robust to differing arguments in scatter vs plot, e.g. "s"/"ms" (marker size)
    if "s" not in kwargs and "ms" in kwargs: kwargs["s"] = kwargs.pop("ms");

    try: 
      classes = np.array(learner.classes);  # learner has explicit list of classes; use those
    except Exception:
        if len(Y.shape)==1 or Y.shape[1]==1:  
            classes = np.unique(Y)                      # or, use data points' class values to guess
        else:
            classes = np.arange(Y.shape[1],dtype=int);  # or, get number of classes from soft predictions
        
    vmin,vmax = classes.min()-.1,classes.max()+.1;      # get (slightly expanded) value range for class values
    if ax is None: ax = plt.gca();                      # default: use current axes
    if cm is None: cm = plt.cm.get_cmap();              # get the colormap
    classvals = (classes-vmin)/(vmax-vmin+1e-100);      # map class values to [0,1] for colormap
    classcolor= cm(classvals);                          # and get the RGB values for each class
    
    ax.plot( X[:,0],X[:,1], 'k.', visible=False, ms=0); # invisible plot to set axis range if required
    axrng = ax.axis();
        
    if learner is not None:                             # if we were given a learner to predict with:
        xticks, yticks = np.linspace(axrng[0],axrng[1],nGrid), np.linspace(axrng[2],axrng[3],nGrid); 
        grid = np.meshgrid( xticks, yticks );           # apply it to a dense grid of points
        XGrid = np.column_stack( (grid[0].flatten(), grid[1].flatten()) )
        if soft: 
            YGrid = learner.predictSoft( pre(XGrid) ).dot(classcolor);     # soft prediction: blend class colors
            YGrid[YGrid<0]=0; YGrid[YGrid>1]=1;
            YGrid = YGrid.reshape((nGrid,nGrid,classcolor.shape[1]))
            #axis.contourf( xticks,yticks,YGrid[:,0].reshape( (len(xticks),len(yticks)) ), nContours )
            ax.imshow( YGrid, extent=axrng, interpolation='nearest',origin='lower',alpha=bgalpha, aspect='auto' , vmin=vmin,vmax=vmax,cmap=cm)
        else:    
            YGrid = learner.predict( pre(XGrid) ).reshape((nGrid,nGrid));  # hard prediction: use class colors
            vmin, vmax = min( YGrid.min()-.1, vmin ), max( YGrid.max()+.1, vmax )     # check outputs for new classes?
            classvals = (classes-vmin)/(vmax-vmin+1e-100); classcolor= cm(classvals); # if so, recalc colors?
            #axis.contourf( xticks,yticks,YGrid.reshape( (len(xticks),len(yticks)) ), nClasses )
            ax.imshow( YGrid, extent=axrng, interpolation='nearest',origin='lower',alpha=bgalpha, aspect='auto' , vmin=vmin,vmax=vmax,cmap=cm)
        
    if len(Y.shape)==1 or Y.shape[1]==1: data_colors = classcolor[np.searchsorted(classes,Y)];  # use colors if Y is discrete class
    else: data_colors = Y.dot(classcolor); data_colors[data_colors>1]=1;             # blend colors if Y is a soft confidence
    ax.scatter(X[:,0],X[:,1], c=data_colors, **kwargs);

    #for i,c in enumerate(classes):                # old code: used plot instead of scatter
    #    ax.plot( X[Y==c,0],X[Y==c,1], 'ko', color=cmap(cvals[i]), **kwargs )


def histy(X,Y,axis=None,**kwargs):
    """
    Plot a histogram (using matplotlib.hist) with multiple classes of data
    Any additional arguments are passed directly into hist()
    Each class of data are plotted as a different color
    To specify specific histogram colors, use e.g. facecolor={0:'blue',1:'green',...}
      so that facecolor[c] is the color for class c
    Related but slightly different appearance to e.g.
      matplotlib.hist( [X[Y==c] for c in np.unique(Y)] , histtype='barstacked' )
    """
    if axis == None: axis = plt
    yvals = np.unique(Y)
    nil, bin_edges = np.histogram(X, **kwargs)
    C,H = len(yvals),len(nil)
    hist = np.zeros( shape=(C,H) )
    cmap = plt.cm.get_cmap()
    cvals = (yvals - min(yvals))/(max(yvals)-min(yvals)+1e-100)
    widthFrac = .25+.75/(1.2+2*np.log10(len(yvals)))
    for i,c in enumerate(yvals):
        histc,nil = np.histogram(X[Y==c],bins=bin_edges)
        hist[i,:] = histc
    for j in range(H):
        for i in np.argsort(hist[:,j])[::-1]:
            delta = bin_edges[j+1]-bin_edges[j]
            axis.bar(bin_edges[j]+delta/2*i/C*widthFrac,hist[i,j],width=delta*widthFrac,color=cmap(cvals[i]))



def plotPairs(X,Y=None,**kwargs):
    """
    Plot all pairs of features in a grid
    Diagonal entries are histograms of each feature
    Off-diagonal are 2D scatterplots of pairs of features
    """
    m,n = X.shape
    if Y is None: Y = np.ones( (m,) )
    fig,ax = plt.subplots(n,n)
    for i in range(n):
        for j in range(n):
            if i == j:
                histy(X[:,i],Y,axis=ax[j,i])
            else:
                plot_classify_2D(None,X[:,[i,j]],Y,axis=ax[j,i])


def plotGauss2D(mu,cov,*args,**kwargs):
    """
    Plot an ellipsoid indicating (one std deviation of) a 2D Gaussian distribution
    All additional arguments are passed into plot(.)
    """
    from scipy.linalg import sqrtm
    theta = np.linspace(0,2*np.pi,50)
    circle = np.array([np.sin(theta),np.cos(theta)])
    ell = sqrtm(cov).dot(circle)
    ell += twod(mu).T

    plt.plot( mu[0],mu[1], 'x', ell[0,:],ell[1,:], **kwargs)





# TODO: plotRegress1D




################################################################################
################################################################################
################################################################################
