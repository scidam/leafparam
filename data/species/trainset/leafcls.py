#coding: utf-8

import numpy as np 
from scipy import interpolate


class LeafOutline(object):

    def __init__(self, contour, resolution=100):
        self.contour = contour
        self.resolution = resolution

    @staticmethod
    def length(contour):
        return np.sum(np.linalg.norm(np.array(contour[1:])-np.array(contour[:-1]),axis=1))
  
    def parametrize():
        _cont = np.array(self.contour)
        tck, u = interpolate.splprep([_cont[:,0], _cont[:,1]], s=0)
        unew = np.arange(0.0, 1.0+1.0/self.resolution, 1.0/self.resolution)
        r = np.array(interpolate.splev(unew, tck)).transpose()
        _l = [length(r[:j]) for j in range(1, len(r)+1)]
        _l = np.array(_l)/np.max(_l)
        ttck = interpolate.splrep(_l, unew)
        regu = interpolate.splev(unew, ttck)
        dr = np.array(interpolate.splev(regu, tck, der=1)).transpose()
        lzero =  dr[:,0] <= 0.0
        _t = np.arctan(dr[:, 1] / dr[:, 0])
        _t[lzero] = np.pi + _t[lzero]
        return (_t.tolist(), unew.tolist(), r)
    



