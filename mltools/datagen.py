import numpy as np

from numpy import loadtxt as loadtxt
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from scipy.linalg import sqrtm



################################################################################
## Methods for creating / sampling synthetic datasets ##########################
################################################################################


def data_gauss(N0, N1=None, mu0=arr([0, 0]), mu1=arr([1, 1]), sig0=np.eye(2), sig1=np.eye(2)):
	"""Sample data from a two-component Gaussian mixture model.  	

	Args:
	    N0 (int): Number of data to sample for class -1.
	    N1 :(int) Number of data to sample for class 1.
	    mu0 (arr): numpy array
	    mu1 (arr): numpy array
	    sig0 (arr): numpy array
	    sig1 (arr): numpy array

	Returns:
	    X (array): Array of sampled data
	    Y (array): Array of class values that correspond to the data points in X.

	TODO: test more
	"""
        # ALT:  return data_GMM_new(N0, ((1.,[0,0],[1.]))
        #       return data_GMM_new(N0+N1, ((.5,[0,0],[1.]),(.5,[1,1],[1.])))
	if not N1:
		N1 = N0

	d1,d2 = twod(mu0).shape[1],twod(mu1).shape[1]
	if d1 != d2 or np.any(twod(sig0).shape != arr([d1, d1])) or np.any(twod(sig1).shape != arr([d1, d1])):
		raise ValueError('data_gauss: dimensions should agree')

	X0 = np.dot(np.random.randn(N0, d1), sqrtm(sig0))
	X0 += np.ones((N0,1)) * mu0
	Y0 = -np.ones(N0)

	X1 = np.dot(np.random.randn(N1, d1), sqrtm(sig1))
	X1 += np.ones((N1,1)) * mu1
	Y1 = np.ones(N1)

	X = np.row_stack((X0,X1))
	Y = np.concatenate((Y0,Y1))

	return X,Y



def gmm_draw_params(m, c, n=2, scale=.05):
    """Create a random Gaussian mixture model.  

    Builds a random GMM with C components and draws M data x^{(i)} from a mixture
    of Gaussians in D dimensions

    Args:
	    m (int): Number of data to be drawn from a mixture of Gaussians.
	    c (int): Number of clusters.
	    n (int): Number of dimensions.
            scale (float): relative scale of the inter- to intra-cluster variance (small = clumpy)

    Returns:
       tuple of tuples, (pi,mu,sig) = mixture weight, mean, and covariance of each component
    """
    pi = np.zeros(c)
    for cc in range(c): pi[cc] = gamrand(10, 0.5)
    pi = pi / np.sum(pi)
    
    rho = np.random.rand(n, n)
    rho = rho + twod(rho).T
    rho = rho + n * np.eye(n)
    rho = sqrtm(rho)
	
    mu = np.random.randn(c, n).dot(rho)

    ccov = []
    for i in range(c):
        tmp = np.random.rand(n, n)
        tmp = tmp + tmp.T
        tmp = scale * (tmp + n * np.eye(n))
        ccov.append(tmp)
    #print(pi,mu,ccov)

    return tuple( (pi[cc],mu[cc],ccov[cc]) for cc in range(c) )


def gmm_draw_samples(M, mixture ):
    """Sample from a Gaussian mixture model.
      Args:
        M (int) : number of samples to draw
        mixture : tuple of tuples, (pi,mu,sig)
      Returns:
        X       : (M x D) data array of samples
        Z       : (M,) array of mixture component ids
    """
    C = len(mixture);
    D = len(mixture[0][1]);
    cpi= np.cumsum([comp[0] for comp in mixture]); cpi /= cpi[-1]
    mu = np.array([comp[1] for comp in mixture]).astype(float)
    ccov=[sqrtm(np.array(comp[2])) for comp in mixture]
    p = np.random.rand(M)
    Z = np.zeros(M, dtype=int)
    for c in range(1,C): Z[p > cpi[c-1]] = c
    X = mu[Z,:]
    for c in range(C): X[Z==c,:] += np.random.randn(np.sum(Z==c), D).dot(ccov[c])
    return X,Z


def gmm_nll(X,mixture):
    """Return the negative log-likelihood of a Gaussian mixture model defined by 'mixture' at X"""
    m,n = X.shape; C = len(mixture);
    pX = np.zeros((m,C))
    for c in range(C):
        pi,mu,sig = mixture[c]; mu = mu.reshape(1,n);
        pX[:,c] = np.log(pi*.5/np.pi/np.linalg.det(sig)**.5) - .5*(((X-mu).dot(np.linalg.inv(sig)))*(X-mu)).sum(1)
    R = pX.max(1,keepdims=True); pX -= R; pX = np.log(np.exp(pX).sum(1)); pX += R.reshape((m,));
    return -pX;

def gmm_nll_dX(X,mixture):
    """Return gradient wrt X of the negative log-likelihood of a Gaussian mixture model defined by 'mixture' at X"""
    m,n = X.shape; C = len(mixture);
    pXc,dX = np.zeros((m,C)), np.zeros((m,n));
    for c in range(C):
        pi,mu,sig = mixture[c]; mu = mu.reshape(1,n);
        pXc[:,c] = np.log(pi*.5/np.pi/np.linalg.det(sig)**.5) - .5*(((X-mu).dot(np.linalg.inv(sig)))*(X-mu)).sum(1)
    R = pXc.max(1,keepdims=True); pXc -= R; pX = np.log(np.exp(pXc).sum(1)); pX += R.reshape((m,));
    for c in range(C):
        pi,mu,sig = mixture[c]; mu = mu.reshape(1,n);
        dX += np.exp(pXc[:,c]-pX).reshape(m,1)*(-1)*(X-mu).dot(np.linalg.inv(sig))
    return -dX;


def data_GMM(m, c, n=2, scale=.05, get_Z=False):
	"""Sample data from a Gaussian mixture model.  

  Builds a random GMM with C components and draws M data x^{(i)} from a mixture
	of Gaussians in D dimensions

	Args:
	    m (int): Number of data to be drawn from a mixture of Gaussians.
	    c (int): Number of clusters.
	    n (int): Number of dimensions.
            scale (float): relative scale of the inter- to intra-cluster variance (small = clumpy)
	    get_Z (bool): If True, returns a an array indicating the cluster from which each 
		    data point was drawn.

	Returns:
	    X (arr): (m,n) nparray of data.
	    Z (arr): (n, ) nparray of cluster ids; returned also only if get_Z=True
    
	TODO: test more; N vs M
	"""
	pi = np.zeros(c+1)
	for cc in range(c+1):
		pi[cc] = gamrand(10, 0.5)
	pi = pi / np.sum(pi)
	cpi = np.cumsum(pi)
	#print(cpi);

	rho = np.random.rand(n, n)
	rho = rho + twod(rho).T
	rho = rho + n * np.eye(n)
	rho = sqrtm(rho)
	
	mu = np.random.randn(c, n).dot(rho)

	ccov = []
	for i in range(c):
		tmp = np.random.rand(n, n)
		tmp = tmp + tmp.T
		tmp = scale * (tmp + n * np.eye(n))
		ccov.append(sqrtm(tmp))
	#print(mu); print(ccov);

	p = np.random.rand(m)
	Z = np.ones(m)

	for cc in range(c):
		Z[p > cpi[cc]] = cc
	Z = Z.astype(int)

	X = mu[Z,:]

	for cc in range(c):
		X[Z == cc,:] += np.random.randn(np.sum(Z == cc), n).dot(ccov[cc])

	if get_Z:
		return (X,Z)
	else:
		return X


def gamrand(alpha, lmbda):
	"""Gamma(alpha, lmbda) generator using the Marsaglia and Tsang method

	Args:
	    alpha (float): scalar
	    lambda (float): scalar
	
	Returns:
	    (float) : scalar

	TODO: test more
	"""
  # (algorithm 4.33).
	if alpha > 1:
		d = alpha - 1./3.
		c = 1./np.sqrt(9 * d)
		flag = 1

		while flag:
			Z = np.random.randn()	

			if Z > -1./c:
				V = (1.+c*Z)**3
				U = np.random.rand()
				flag = np.log(U) > (0.5 * Z**2 + d - d * V + d * np.log(V))

		return d * V / lmbda

	else:
		x = gamrand(alpha + 1, lmbda)
		return x * np.random.rand()**(1./alpha)


## OLD VERSION
def data_mouse(fig=None):
	"""Simple by-hand data generation using the GUI

	Opens a matplotlib plot window, and allows the user to specify points with the mouse.
	Each button is its own class (1,2,3); close the window when done creating data.

  Returns:
      X (arr): Mx2 array of data locations
      Y (arr): Mx1 array of labels (buttons)
	"""
	import matplotlib.pyplot as plt
	if fig is None:
		fig = plt.figure()
		ax = fig.add_subplot(111, xlim=(-1,2), ylim=(-1,2))
	else:
		ax = fig.gca()
	X  = np.zeros( (0,2) )
	Y  = np.zeros( (0,) )
	col = ['bs','gx','ro']
	
	def on_click(event):
		X.resize( (X.shape[0]+1,X.shape[1]) )
		X[-1,:] = [event.xdata,event.ydata]
		Y.resize( (Y.shape[0]+1,) )
		Y[-1] = event.button
		ax.plot( event.xdata, event.ydata, col[event.button-1])
		fig.canvas.draw()

	fig.canvas.mpl_connect('button_press_event',on_click)
	plt.show()
	return X,Y


################ New version ################################

class MouseData(object):
    """Object to create & store data from mouse input 
        Can use interactive plotting (%matplotlib notebook) or ipywidgets+ipevents (%matplotlib inline)
        Use mouse click to add data points (hold combos of shift,alt,ctrl to set target value)
        Use right click to remove data points
    """
    def update_plot(self): pass
    def __update_data(self,event): pass

    @property
    def X(self):
        """Access the current list of (2d) locations (feature values)"""
        return self.XX[:self.__m,:]
    @property
    def Y(self):
        """Access the current list of target values"""
        return self.YY[:self.__m]
    @property
    def m(self): return self.__m

    def add_point(self,x,y):
        """Add point x=[x1,x2] and target value y to the data set"""
        self.XX[self.__m,:] = x; 
        self.YY[self.__m] = y; 
        self.__m += 1;

    def remove_nearest(self,x):
        """Remove the point closest to x from the data set"""
        idx = ((self.XX[:self.__m,:]-x)**2).sum(1).argmin()   # TODO: check shape; when X is 2x2?
        self.XX[idx,:] = self.XX[self.__m-1,:]; self.YY[idx]=self.YY[self.__m-1];
        self.__m -= 1;

    def __init__(self, m=200, figsize=(6,6), plot=None):
        self.border = .01
        self.__m = 0;
        self.XX = np.zeros((m,2));
        self.YY = np.zeros((m,))-1;
        self.plot = plot;
        self.lim = [-1,1,-1,1];

    def __repr__(self):
        return self.image
    def __str__(self):
        return "Mouse-based data input object; {} data ({} maximum)\n".format(self.m,len(self.XX))+self.image.__str__()

# MPL version
keys = {'none':0, 'shift':1, 'control':2, 'ctrl+shift':3,
        'alt':4, 'alt+shift':5, 'alt+control':6, 'ctrl+alt+shift':7}
# IPyWidget version
def wkeys(event): return int(event['shiftKey']+2*event['ctrlKey']+4*event['altKey']);

class MouseDataPlot(MouseData):
    """Get data from mouse input, using standard pyplot methods (Jupyter: %matplotlib notebook)
        usage:  data = MouseDataPlot(); data.display(); ...
        after input, stores 'data.m' points, with locations 'data.X' and targets 'data.Y'
    """
    def __clear(self):
        import matplotlib.pyplot as plt
        plt.cla()
        self.fig.add_axes((self.border,self.border,1-2*self.border,1-2*self.border))  # full width figure
        self.ax = self.fig.gca(); 
        self.ax.axis(self.lim);        
        self.ax.set_xticks([]); self.ax.set_yticks([]);

    def display(self, fig=None, **kwargs):
        import matplotlib.pyplot as plt
        self.border = .01
        if fig is None:
            self.fig=plt.figure(figsize=(6,6));   # Create & save a figure; we will re-draw the canvas
            self.__clear()
        else:
            self.fig = fig                        # User-provided figure handle; get axis & plot range
            self.ax  = fig.gca()
            self.lim = ax.axes()
        if self.plot is not None: self.plot(self.X,self.Y)
        
        def on_click(event):
            if (event.button == 1):     # left click: add data point
                self.add_point( [event.xdata,event.ydata], keys.get(event.key,0) )
            elif (event.button == 3):   # right click: remove data point
                self.remove_nearest( [event.xdata,event.ydata] )
            else: pass
            self.__clear()              # clear figure & redraw
            if self.plot is not None: self.plot(self.X,self.Y)       # user-provided plot f'n
            else: self.ax.scatter(self.X[:,0],self.X[:,1], c=self.Y) # or default
            self.fig.canvas.draw()

        self.fig.canvas.mpl_connect('button_press_event',on_click)
        plt.show()


blank_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x03\x00\x08\xfc\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82'

def plot_to_png(fig=None):
    """Convert pyplot figure to ipywidget Image object (or data for same), for interactive display"""
    import matplotlib.pyplot as plt
    import io
    if fig is None: fig=plt.gcf();
    fig.canvas.draw();
    with io.BytesIO() as output:
        fig.canvas.print_png(output)
        contents = output.getvalue()
    return contents


class MouseDataWidget(MouseData):
    """Get data from mouse input, using ipywidget/ipyevent methods (Jupyter: %matplotlib inline)
        usage:  data = MouseDataWidget(); data.display(); ...
        after input, stores 'data.m' points, with locations 'data.X' and targets 'data.Y'
        plot image widget available as "data.image"
    """
    def __init__(self, *args, **kwargs):
        from ipywidgets import Image
        from ipyevents import Event
        #self.plot = None   # set in super?
        super().__init__(*args,**kwargs);
        self.image = Image(value=blank_png, format='png');
        self.no_drag = Event(source=self.image, watched_events=['dragstart'], prevent_default_action = True)
        self.events = Event(source=self.image, watched_events=['click','contextmenu'], prevent_default_action=True)
        self.events.on_dom_event(self.__on_click)
        self.update_plot();

    def __get_coord(self,event):
        return  np.array([event['dataX']/self.w,1-event['dataY']/self.h])*2/(1-2*self.border) - 1-self.border

    def update_plot(self):
        import matplotlib.pyplot as plt
        from .plot import plotClassify2D
        fig=plt.figure(figsize=(6,6))
        plt.gcf().add_axes((self.border,self.border,1-2*self.border,1-2*self.border))  # full width figure
        plt.axis(self.lim); plt.gca().set_xticks([]); plt.gca().set_yticks([]);
        if (self.m > 0) or (self.plot is not None):
            if self.plot is not None: self.plot(self.X,self.Y)       # user-provided plot f'n
            else: fig.gca().scatter(self.X[:,0],self.X[:,1], c=self.Y) # or default
        self.image.value = plot_to_png(plt.gcf());
        self.w,self.h = plt.gcf().canvas.get_renderer().get_canvas_width_height();
        fig.clear();

    def display(self, fig=None, **kwargs):
        from IPython.display import display
        if self.plot is None: print("Default plot. Left-click to add data; shift/alt/ctrl combos determine target value; right click to remove data.")
        display(self.image)     
        
    def __on_click(self,event):
        if (event['type']=='click'):          # left click = add point
            self.add_point( self.__get_coord(event), wkeys(event) )
        elif (event['type']=='contextmenu') and (self.m>0):  # right click = remove point
            self.remove_nearest( self.__get_coord(event) )
        self.update_plot()

