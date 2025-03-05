# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:12:04 2020

@author: Francien
"""
import numpy as np
from scipy.ndimage import gaussian_filter

#%%
def phantom_plank(phantom_size, angle = 0, p = 1, v0 = None, ring_width = 1, y1 = 1, y2 = 2, gaussian_blurr = 0): 
    '''
    Parameters
    ----------
    t : thickness of the plank (in mm)
    w : width of the plank (in mm)
    l : length of the plank (in mm)
    vx_00 : x location of the left upper pixel relative to the pith (in mm)
    vy_00 : y location of the left upper pixel relative to the pith (in mm)
    vz_00 : z location of the left upper pixel relative to the pith (in mm)
    To have pith in middle of phantom: 
        v00 = (-1/2*phantom_size[1], 1/2*phantom_size[0],1/2*phantom_size[2])
    angle : tilt of the pith, in radians
    p : pixel size. The default is 1.
    y1 : lowest density value. The default is 1.
    y2 : highest density value. The default is 2. 
    
    Returns
    -------
    phantom : 3D plank phantom, with tree rings. 

    '''
    #If there are no explicit numbers for a,b,c,d,e, but a period given,
    #calculate the numbers a to e as ratio's of the period. 
    t, l, w = phantom_size
    if v0 is not None: 
        vx_00, vy_00, vz_00 = v0
    else: 
        vx_00, vy_00, vz_00 = (-phantom_size[2],phantom_size[0]/2,0)
    #period = np.ceil(ring_width/p)
    period = ring_width
    #period>0.5
    a = 0
    b = 0.2#4/10*period
    c = 1/2*period
    d = 1/2*period+0.2#(9/10)*period
    e = period
    
    
    
    m = int(np.ceil(t/p))
    n = int(np.ceil(w/p))
    o = int(np.ceil(l/p))
        
    phantom = np.zeros((m,n,o), dtype = 'float32')
    
    #If the tree is straight, every layer is the same. 
    if angle == 0:
        for i in range(m):
            for j in range(n):
                dist = distance_to_pith(i, j, 0, vx_00,vy_00, 0, 0, p)
                value = truncated_triangle(dist, y1, y2, a, b, c, d, e)
                phantom[i,j,:] = value
    
    #If the tree is at an angle, or the plank is cut at an angle, 
                #every layer is different. 
    else: 
        for i in range(m):
                for j in range(n):
                    for k in range(o):
                        dist = distance_to_pith(i, j, k, vx_00, vy_00, vz_00, angle, p)
                        value = truncated_triangle(dist, y1, y2, a, b, c, d, e)
                        phantom[i,j,k] = value
    
    #plt.imshow(phantom[:,:,0])
    if gaussian_blurr>0: 
        for i in np.arange(phantom.shape[2]):
            phantom[:,:,i] = gaussian_filter(phantom[:,:,i],sigma = gaussian_blurr)
    
    return phantom

def distance_to_pith(r, c, lr, vx_00, vy_00, vz_00, angle, p):
    '''

    Parameters
    ----------
    r : rownumber
    c : columnnumber
    lr: layernumber (slice in z-direction, which is along the tree)
    vx_00 : x location of the left upper pixel relative to the pith in mm
    vy_00 : y location of the left upper pixel relative to the pith in mm
    vz_00 : z location of the left upper pixel relative to the pith in mm
    p : pixel size, optional, The default is 1.
    

    Returns
    -------
    distance from pith of tree to pixel

    '''
    vx = vx_00 + p*c
    vy = vy_00 - p*r
    vz = vz_00 - p*lr
    
    return np.sqrt(vx**2 + vy**2) + vz * np.arcsin(angle) 

def truncated_triangle(l, y1, y2, a, b, c, d, e):
    '''
    Parameters
    ----------
    l : distance to pith of tree 
    y1 : lowest function value
    y2 : highest function value
    a : start of period including two tree rings, one year: 
        a denser ring and a less dense ring, start of increasing slope
    b : stop of the increasing slope
    c : start of the decreasing slope
    d : stop of the decreasing slope
    e : end of period

    Returns
    -------
    Value for the pixel in the tree phantom, based on a truncated triangle 
    function. 

    '''
    l_mod = np.mod(l,e-a)
    
    if a <= l_mod  < b: 
        return np.abs((y2-y1)/(b-a)*(l_mod-a)+y1)
    elif b <= l_mod < c:
        return y2
    elif c <= l_mod < d:
        return np.abs(-(y2-y1)/(d-c)*(l_mod-c)+y2)
    elif d <= l_mod < e:
        return y1


