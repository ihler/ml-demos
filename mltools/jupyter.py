################################################################################
## IMPORTS #####################################################################
################################################################################


import numpy as np
from ipywidgets import Label, HTML, HBox, Image, VBox, Box, HBox
from ipyevents import Event
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
import io

from numpy import asarray as arr
from numpy import atleast_2d as twod

from .plot import plotClassify2D


blank_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x03\x00\x08\xfc\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82'

def plot_to_widget(fig=None, data_only=False):
    """Convert pyplot figure to ipywidget Image object (or data for same), for interactive display"""
    #plt.tight_layout()
    if fig is None: fig=plt.gcf();
    fig.canvas.draw();
    with io.BytesIO() as output:
        fig.canvas.print_png(output)
        contents = output.getvalue()
    if data_only:
        return contents
    else:
        return Image(value=contents, format='png')



class data_mouse(object):
    """Object to create & store data from mouse input in (ipywidget+ipyevents) jupyter notebook
        Use mouse click to add data points (hold combos of shift,alt,ctrl to set target value)
        Use right click to remove data points
    """
    def __update_plot(self):
        fig=plt.figure(figsize=(6,6))
        plt.gcf().add_axes((self.border,self.border,1-2*self.border,1-2*self.border))  # full width figure
        plt.axis(self.ax); plt.gca().set_xticks([]); plt.gca().set_yticks([]);
        if (self.m > 0) or (self.plot is not None):
            if (self.plot is not None):
                self.plot(self.X,self.Y)   # call passed function handle for data plot
                #plotClassify2D(self.learner(self.X,self.Y), self.XX[:self.m,:],self.YY[:self.m]) # add options?
            else:
                plotClassify2D(None, self.XX[:self.m,:],self.YY[:self.m]) # add options?
        self.image.value = plot_to_widget(plt.gcf(), data_only=True);
        self.w,self.h = plt.gcf().canvas.get_renderer().get_canvas_width_height();
        #_=plt.clf();
        fig.clear();

    def __update_data(self,event):
        if (event['type']=='click'):          # left click = add point
            #self.XX[self.m,:] = np.array([event['dataX'],-event['dataY']])/200. + [-1.1,1.1];
            self.XX[self.m,:] = np.array([event['dataX']/self.w,1-event['dataY']/self.h])*2/(1-2*self.border) - 1-self.border
            self.YY[self.m] = int(event['shiftKey']+2*event['ctrlKey']+4*event['altKey']);
            self.M[0] += 1 
        elif (event['type']=='contextmenu') and (self.m>0):  # right click = remove point
            xrem = np.array([event['dataX']/self.w,1-event['dataY']/self.h])*2/(1-2*self.border) - 1-self.border
            idx = ((self.XX[:self.m,:] - xrem)**2).sum(1).argmin()
            self.XX[idx,:] = self.XX[self.m-1,:]; self.YY[idx] = self.YY[self.m-1]
            self.M[0] -= 1
        self.__update_plot()

    @property
    def X(self):
        return self.XX[:self.m,:]
    @property
    def Y(self):
        return self.YY[:self.m]
    @property
    def m(self): return self.M[0]

    def __init__(self, m=200, figsize=(6,6), plot=None):
        self.border = .01
        self.M = np.array([0]);
        self.XX = np.zeros((m,2));
        self.YY = np.zeros((m,))-1;
        self.plot = plot;
        self.ax = [-1,1,-1,1];
        self.image = Image(value=blank_png,format='png');
        self.__update_plot();

        self.no_drag = Event(source=self.image, watched_events=['dragstart'], prevent_default_action = True)
        #self.events = Event(source=self.image, watched_events=['click'])
        self.events = Event(source=self.image, watched_events=['click','contextmenu'], prevent_default_action=True)
        self.events.on_dom_event(self.__update_data)
        if self.plot is None: print("Default plot. Left-click to add data; shift/alt/ctrl combos determine target value; right click to remove data.")

    def __repr__(self):
        return repr(self.image)
    def _ipython_display_(self):
        return self.image._ipython_display_()
    def __str__(self):
        return "Mouse-based data input object; {} data ({} maximum)\n"+self.image.__str__()

