#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:53:07 2020

@author: mohammad.mirkazemi
"""
from typing import List
import numpy as np
from jax import numpy as jnp
from ._settings import settings
from .tools._resize import _resize_images_list, _resize_image
from .tools._dct2d_tools import dct2d
from .tools.inexact_alm_rspca_l1 import inexact_alm_rspca_l1

def background_timelapse(
        images_list: List,
        flatfield: jnp.ndarray = None,
        darkfield: jnp.ndarray = None,
        verbosity = True,
        **kwargs
        ):

    """
    Computes the baseline drift for the input images and returns a numpy 1D array

    Parameters:
    ----------
    images_list : list
        A list of 2D arrays as the list of input images. The list can be provided by 
        using pybasic.load_data() function.
        
    flatfield : numpy 2D array
        A flatfield image for input images with the same shape as them. The flatfield 
        image may be calculated using pybasic.basic() function.
        
    darkfield : numpy 2D array, optional
        A darkfield image for input images with the same shape as them. The darkfield 
        image may be calculated using the `pybasic.basic()` function.
        
    verbosity : Boolean
        If True the reweighting iteration number is printed (default is True).  

    Returns:
    --------
        A 1d numpy array containing baseline drift for each input image. The length of 
        the array equals the length of the list of input images. 
            
    """
    
    for _key, _value in kwargs.items():
        setattr(settings, _key, _value)

    nrows = ncols = _working_size = settings.working_size

    # Reszing
    # cv2.INTER_LINEAR is not exactly the same method as 'bilinear' in MATLAB
    
    resized_images = jnp.stack(_resize_images_list(images_list=images_list, side_size=_working_size))
    resized_images = resized_images.reshape([-1, nrows * nrows], order = 'F')

    resized_flatfield = _resize_image(image = flatfield, side_size = _working_size)
    
    if darkfield is not None:
        resized_darkfield = _resize_image(image = darkfield, side_size = _working_size)
    else:
        resized_darkfield = jnp.zeros(resized_flatfield.shape, jnp.uint8)
            
    _weights = jnp.ones(resized_images.shape)
    eplson = 0.1
    tol = 1e-6
    for reweighting_iter in range(1,6):
        W_idct_hat = jnp.reshape(resized_flatfield, (1,-1), order='F')
        A_offset = jnp.reshape(resized_darkfield, (1,-1), order='F')
        A1_coeff = jnp.mean(resized_images, 1).reshape([-1,1])

        # main iteration loop starts:
        # The first element of the second array of jnp.linalg.svd
        _temp = jnp.linalg.svd(resized_images, full_matrices=False)[1]
        norm_two = _temp[0]

        mu = 12.5/norm_two # this one can be tuned
        mu_bar = mu * 1e7
        rho = 1.5 # this one can be tuned
        d_norm = jnp.linalg.norm(resized_images, ord = 'fro')
        ent1 = 1
        _iter = 0
        total_svd = 0
        converged = False;
        A1_hat = jnp.zeros(resized_images.shape)
        E1_hat = jnp.zeros(resized_images.shape)
        Y1 = 0
            
        while not converged:
            _iter = _iter + 1;
            A1_hat = W_idct_hat * A1_coeff + A_offset

            # update E1 using l0 norm
            E1_hat = E1_hat + jnp.divide((resized_images - A1_hat - E1_hat + (1/mu)*Y1), ent1)
            E1_hat = jnp.maximum(E1_hat - _weights/(ent1*mu), 0) +\
                     jnp.minimum(E1_hat + _weights/(ent1*mu), 0)
            # update A1_coeff, A2_coeff and A_offset
            #if coeff_flag
            
            R1 = resized_images - E1_hat
            A1_coeff = jnp.mean(R1,1).reshape(-1,1) - jnp.mean(A_offset,1)

            A1_coeff[A1_coeff<0] = 0
                
            Z1 = resized_images - A1_hat - E1_hat

            Y1 = Y1 + mu*Z1

            mu = min(mu*rho, mu_bar)
                
            # stop Criterion  
            stopCriterion = jnp.linalg.norm(Z1, ord = 'fro') / d_norm
            # print(stopCriterion, tol)
            if stopCriterion < tol:
                converged = True
            # if total_svd % 10 == 0:
            #     print('stop')
                
        # updating weight
        # XE_norm = E1_hat / jnp.mean(A1_hat)
        XE_norm = E1_hat
        mean_vec = jnp.mean(A1_hat, axis=1)
        if verbosity:
            print("reweighting_iter:", reweighting_iter)
        XE_norm = jnp.transpose(jnp.tile(mean_vec, (nrows * nrows, 1))) / (XE_norm + 1e-6)
        _weights = 1./(abs(XE_norm)+eplson)

        _weights = jnp.divide( jnp.multiply(_weights, _weights.shape[0] * _weights.shape[1]), jnp.sum(_weights))

    return jnp.squeeze(A1_coeff) 


def basic(images_list: List, segmentation: List = None, verbosity = True, **kwargs):
    """
    Computes the illumination background for a list of input images and returns flatfield 
    and darkfield images. The input images should be monochromatic and multi-channel images 
    should be separated, and each channel corrected separately.

    Parameters:
    ----------
    images_list : list
        A list of 2D arrays as the list of input images. The list may be provided by u
        sing the `pybasic.load_data()` function.
        
    darkfield : boolean
        If True then darkfield is also computed (default is False).
        
    verbosity : Boolean
        If True the reweighting iteration number is printed (default is True).  

    Returns:
    --------
    flatfield : numpy 2D array
        Flatfield image of the calculated illumination with the same size of input numpy arrays.
        
    darkfield : numpy 2D array
        Darkfield image of the calculated illumination with the same size of input numpy array. 
        If the darkfield argument of the function is set to False, then an array of zeros with 
        the same shape of input arrays is returned.
    """
    for _key, _value in kwargs.items():
        setattr(settings, _key, _value)

    nrows = ncols = _working_size = settings.working_size
    
    _saved_size = images_list[0].shape

    D = jnp.dstack(_resize_images_list(images_list=images_list, side_size=_working_size))

    meanD = jnp.mean(D, axis=2)
    meanD = meanD / jnp.mean(meanD)
    W_meanD = dct2d(np.array(meanD.T))
    if settings.lambda_flatfield == 0:
        setattr(settings, 'lambda_flatfield', jnp.sum(jnp.abs(W_meanD)) / 400 * 0.5)
    if settings.lambda_darkfield == 0:
        setattr(settings, 'lambda_darkfield', settings.lambda_flatfield * 0.2)

    # TODO: Ask Tingying whether to keep sorting? I remember the sorting caused some problems with some data.
    D = jnp.sort(D, axis=2)

    XAoffset = jnp.zeros((nrows, ncols))
    weight = jnp.ones(D.shape)

    if segmentation is not None:
        segmentation = jnp.array(segmentation)
        segmentation = jnp.transpose(segmentation, (1, 2, 0))
        for i in range(weight.shape[2]):
            weight[segmentation] = 1e-6
        # weight[options.segmentation] = 1e-6

    reweighting_iter = 0
    flag_reweighting = True
    flatfield_last = jnp.ones((nrows, ncols))
    darkfield_last = np.random.randn(nrows, ncols)

    while flag_reweighting:
        reweighting_iter += 1
        if verbosity:
            print("reweighting_iter:", reweighting_iter)
        initial_flatfield = False
        if initial_flatfield:
            # TODO: implement inexact_alm_rspca_l1_intflat?
            raise IOError('Initial flatfield option not implemented yet!')
        else:
            X_k_A, X_k_E, X_k_Aoffset = inexact_alm_rspca_l1(D, weight=weight);
        XA = jnp.reshape(X_k_A, [nrows, ncols, -1], order='F')
        XE = jnp.reshape(X_k_E, [nrows, ncols, -1], order='F')
        XAoffset = jnp.reshape(X_k_Aoffset, [nrows, ncols], order='F')
        XE_norm = XE / jnp.mean(XA, axis=(0, 1))

        # Update the weights:
        weight = jnp.ones_like(XE_norm) / (jnp.abs(XE_norm) + settings.eplson)
        if segmentation is not None:
            weight[segmentation] = 0

        weight = weight * weight.size / jnp.sum(weight)

        temp = jnp.mean(XA, axis=2) - XAoffset
        flatfield_current = temp / jnp.mean(temp)
        darkfield_current = XAoffset
        mad_flatfield = jnp.sum(jnp.abs(flatfield_current - flatfield_last)) / jnp.sum(jnp.abs(flatfield_last))
        temp_diff = jnp.sum(jnp.abs(darkfield_current - darkfield_last))
        if temp_diff < 1e-7:
            mad_darkfield = 0
        else:
            mad_darkfield = temp_diff / jnp.maximum(jnp.sum(jnp.abs(darkfield_last)), 1e-6)
        flatfield_last = flatfield_current
        darkfield_last = darkfield_current
        if jnp.maximum(mad_flatfield,
                      mad_darkfield) <= settings.reweight_tolerance or \
                reweighting_iter >= settings.max_reweight_iterations:
            flag_reweighting = False

    shading = jnp.mean(XA, 2) - XAoffset

    flatfield = _resize_image(
        image = shading, 
        x_side_size = _saved_size[0], 
        y_side_size = _saved_size[1]
    )
    flatfield = flatfield / jnp.mean(flatfield)

    if settings.darkfield:
        darkfield = _resize_image(
            image = XAoffset, 
            x_side_size = _saved_size[0], 
            y_side_size = _saved_size[1]
        )
    else:
        darkfield = jnp.zeros_like(flatfield)

    return flatfield, darkfield

def correct_illumination(
    images_list: List, 
    flatfield: jnp.ndarray = None, 
    darkfield: jnp.ndarray = None,
    background_timelapse: jnp.ndarray = None,
):
    """
    Applies the illumination correction on a list of input images 
    and returns a list of corrected images.

    Parameters
    ----------
    images_list : list

        A list of 2D arrays as the list of input images. The list can be provided by using pybasic.load_data() function.
        
    flatfield : numpy 2D array

        A flatfield image for input images with the same shape as them. The flatfield image may be calculated using pybasic.basic() function.
        
    darkfield : numpy 2D array, optional

        A darkfield image for input images with the same shape as them. The darkfield image may be calculated using the `pybasic.basic()` function.

    background_timelapse : numpy 1D array or a list, optional
        Timelapse background or baseline drift of the images in the same order as images in the input list. The lenght of background_timelapse should be as the same as the length of list of input images.


    Returns:
    --------
        A list of illumination corrected images with the same length of list of input images.
    """

    _saved_size = images_list[0].shape
    
    if not flatfield.shape == _saved_size:
        flatfield = _resize_image(
            image = flatfield, 
            x_side_size = _saved_size[0], 
            y_side_size = _saved_size[1]
        )
    
    if darkfield is None:
        corrected_images = [_im / flatfield for _im in images_list]
    else:
        if not darkfield.shape == _saved_size:
            darkfield = _resize_image(
                image = darkfield, 
                x_side_size = _saved_size[0], 
                y_side_size = _saved_size[1]
            )
        corrected_images = [(_im  - darkfield)/ flatfield for _im in images_list]

    if background_timelapse is not None:
        if len(background_timelapse) != len(corrected_images):
            print(f"Error: background_timelapse and input images should have the same lenght.")
        for i, bg in enumerate(background_timelapse):
            corrected_images[i] = corrected_images[i] - bg

    return corrected_images
