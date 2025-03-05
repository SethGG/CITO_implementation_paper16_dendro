#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 08:21:03 2020

@author: bossema
"""
import numpy as np
from flexdata import geometry
from flexdata import display

#%%
def line_geometry(sod, odd, det_pixel, vox_size, source_range, det_range, plot = True):
    geom = geometry.linear(src2obj = sod, det2obj =odd, det_pixel = det_pixel, img_pixel = vox_size, 
                      src_hrz_rng = source_range, src_vrt_rng = (0, 0), det_hrz_rng = det_range, det_vrt_rng = (0,0))
    if plot is True:
            orbit = geom.get_source_orbit(50)
            display.plot3d(orbit[:, 0], orbit[:, 1], orbit[:, 2], connected = True, title = 'Linear source orbit')
    return geom

def system_details(geom, obj_size, det_width):
    sod = geom['src2obj']
    odd = geom['det2obj']
    det_pixel = geom['det_pixel']
    
    mag = (sod+odd)/sod
    print('magnification:', mag )
    max_angle = np.arctan(det_width*det_pixel/(2*(sod+odd)))
    print('maximum angle:', np.rad2deg(max_angle))
    
    sod_new = sod-obj_size[1]/2
    min_increment = det_pixel*(sod+odd)/(sod_new)
    min_proj = abs(geom['src_hrz_rng'][1]-geom['src_hrz_rng'][0])/min_increment
    print('minimum increment', min_increment, '\nmin_proj', min_proj)
    return min_proj
    
def min_range(phantom_size, angle, sod): 
    p_width = phantom_size[1]
    p_length = phantom_size[2] 

    if angle >= 0:
        left_range = 1/2*p_width + 1/2*p_length*np.tan(angle) + sod*np.tan(angle)
        right_range = -1/2*p_width - 1/2*p_length*np.tan(angle)
        print('necessary range:', left_range, right_range)
    
    if angle < 0:
        left_range = 1/2*p_width - 1/2*p_length*np.tan(angle)
        right_range = -1/2*p_width + 1/2*p_length*np.tan(angle) + sod*np.tan(angle)
        print('necessary range:', left_range, right_range)

    return (left_range,right_range)

def safe_range(phantom_size, angle, sod):
    p_width = phantom_size[1]
    p_length = phantom_size[2] 
    
    left_range = 1/2*p_width + 1/2*p_length*np.tan(angle) + sod*np.tan(angle)
    right_range = -left_range
    print('necessary range:', left_range, right_range)
    
    return (left_range,right_range)
