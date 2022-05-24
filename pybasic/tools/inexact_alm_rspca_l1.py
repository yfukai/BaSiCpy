#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:04:46 2020

@author: mohammad.mirkazemi
"""


import numpy as np
from jax import numpy as jnp
from .._settings import settings
from ..tools._dct2d_tools import dct2d, idct2d


def _shrinkageOperator(matrix, epsilon):
    return jnp.sign(matrix) * jnp.maximum(jnp.abs(matrix) - epsilon, 0)


def inexact_alm_rspca_l1(images, weight=None, **kwargs):
    for _key, _value in kwargs.items():
        setattr(settings, _key, _value)

    if weight is not None and weight.size != images.size:
            raise IOError('weight matrix has different size than input sequence')

    # if 
    # Initialization and given default variables
    p, q, n = images.shape
    m = p*q
    images = jnp.reshape(images, (m, n), order='F')

    if weight is not None:
        weight = jnp.reshape(weight, (m, n), order='F')
    else:
        weight = jnp.ones_like(images,dtype=np.float32)
    svd = jnp.linalg.svd(images, False, False) #TODO: Is there a more efficient implementation of SVD?
    norm_two = svd[0]
    Y1 = 0
    #Y2 = 0
    ent1 = 1
    ent2 = 10

    A1_hat = jnp.zeros_like(images,dtype=np.float32)
    A1_coeff = jnp.ones((1, images.shape[1]))

    E1_hat = jnp.zeros_like(images,dtype=np.float32)
    W_hat = dct2d(np.array(jnp.zeros((p, q),dtype=np.float32).T))
    mu = 12.5 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = jnp.linalg.norm(images, ord='fro')

    A_offset = jnp.zeros((m, 1),dtype=np.float32)
    B1_uplimit = jnp.min(images)
    B1_offset = 0
    #A_uplimit = jnp.expand_dims(jnp.min(images, axis=1), 1)
    A_inmask = np.zeros((p, q),dtype=np.float32)
    A_inmask[int(jnp.round(p / 6) - 1): int(jnp.round(p*5 / 6)), int(jnp.round(q / 6) - 1): int(jnp.round(q * 5 / 6))] = 1

    # main iteration loop starts
    iter = 0
    total_svd = 0
    converged = False

    #time_zero = time.time()
    #time_zero_it = time.time()
    while not converged:
    #    time_zero_it = time.time()
        iter += 1

        if len(A1_coeff.shape) == 1:
            A1_coeff = jnp.expand_dims(A1_coeff, 0)
        if len(A_offset.shape) == 1:
            A_offset = jnp.expand_dims(A_offset, 1)
        W_idct_hat = idct2d(np.array(W_hat.T))
        A1_hat = jnp.dot(jnp.reshape(W_idct_hat, (-1,1), order='F'), A1_coeff) + A_offset

        temp_W = (images - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
        temp_W = jnp.reshape(temp_W, (p, q, n), order='F')
        temp_W = jnp.mean(temp_W, axis=2)
        W_hat = W_hat + dct2d(np.array(temp_W.T))
        W_hat = jnp.maximum(W_hat - settings.lambda_flatfield / (ent1 * mu), 0) + jnp.minimum(W_hat + settings.lambda_flatfield / (ent1 * mu), 0)
        W_idct_hat = idct2d(np.array(W_hat.T))
        if len(A1_coeff.shape) == 1:
            A1_coeff = jnp.expand_dims(A1_coeff, 0)
        if len(A_offset.shape) == 1:
            A_offset = jnp.expand_dims(A_offset, 1)
        A1_hat = jnp.dot(jnp.reshape(W_idct_hat, (-1,1), order='F'), A1_coeff) + A_offset
        E1_hat = images - A1_hat + (1 / mu) * Y1 / ent1
        E1_hat = _shrinkageOperator(E1_hat, weight / (ent1 * mu))
        R1 = images - E1_hat
        A1_coeff = jnp.mean(R1, 0) / jnp.mean(R1)
        A1_coeff.at[A1_coeff < 0].set(0)

        if settings.darkfield:
            validA1coeff_idx = np.where(A1_coeff < 1)
            R1=np.array(R1)
            B1_coeff = (jnp.mean(R1[jnp.reshape(W_idct_hat, -1, order='F') > jnp.mean(W_idct_hat) * (1-1e-3)][:, validA1coeff_idx[0]], 0) - \
            jnp.mean(R1[jnp.reshape(W_idct_hat, -1, order='F') < jnp.mean(W_idct_hat) * (1+1e-3)][:, validA1coeff_idx[0]], 0)) / jnp.mean(R1)
            k = jnp.array(validA1coeff_idx).shape[1]
            A1_coeff = np.array(A1_coeff)
            B1_coeff = np.array(B1_coeff)
            temp1 = jnp.sum(A1_coeff[validA1coeff_idx[0]]**2)
            temp2 = jnp.sum(A1_coeff[validA1coeff_idx[0]])
            temp3 = jnp.sum(B1_coeff)
            temp4 = jnp.sum(A1_coeff[validA1coeff_idx[0]] * B1_coeff)
            temp5 = temp2 * temp3 - temp4 * k
            if temp5 == 0:
                B1_offset = 0
            else:
                B1_offset = (temp1 * temp3 - temp2 * temp4) / temp5
            # limit B1_offset: 0<B1_offset<B1_uplimit

            B1_offset = jnp.maximum(B1_offset, 0)
            B1_offset = jnp.minimum(B1_offset, B1_uplimit / jnp.mean(W_idct_hat))

            B_offset = B1_offset * jnp.reshape(W_idct_hat, -1, order='F') * (-1)

            B_offset = B_offset + jnp.ones_like(B_offset) * B1_offset * jnp.mean(W_idct_hat)
            A1_offset = jnp.mean(R1[:, validA1coeff_idx[0]], axis=1) - jnp.mean(A1_coeff[validA1coeff_idx[0]]) * jnp.reshape(W_idct_hat, -1, order='F')
            A1_offset = A1_offset - jnp.mean(A1_offset)
            A_offset = A1_offset - jnp.mean(A1_offset) - B_offset

            # smooth A_offset
            W_offset = dct2d(np.array(jnp.reshape(A_offset, (p,q), order='F').T))
            W_offset = jnp.maximum(W_offset - settings.lambda_darkfield / (ent2 * mu), 0) + \
                jnp.minimum(W_offset + settings.lambda_darkfield / (ent2 * mu), 0)
            A_offset = idct2d(np.array(W_offset.T))
            A_offset = jnp.reshape(A_offset, -1, order='F')

            # encourage sparse A_offset
            A_offset = jnp.maximum(A_offset - settings.lambda_darkfield / (ent2 * mu), 0) + \
                jnp.minimum(A_offset + settings.lambda_darkfield / (ent2 * mu), 0)
            A_offset = A_offset + B_offset


        Z1 = np.array(images - A1_hat - E1_hat,dtype=np.float32)
        Y1 = np.array(Y1 + mu * Z1,dtype=np.float32)
        mu = jnp.minimum(mu * rho, mu_bar)

        # Stop Criterion
        stopCriterion = jnp.linalg.norm(Z1, ord='fro') / d_norm
        if stopCriterion < settings.optimization_tolerance:
            converged = True
        """
        if total_svd % 10 == 0:
            print('Iteration', iter, ' |W|_0 ', jnp.sum(jnp.abs(W_hat) > 0), '|E1|_0', jnp.sum(jnp.abs(E1_hat) > 0), \
                  ' stopCriterion', stopCriterion, 'B1_offset', B1_offset)
        """
        if not converged and iter >= settings.max_iterations:
            print('Maximum iterations reached')
            converged = True

    A_offset = jnp.squeeze(A_offset)
    A_offset = A_offset + B1_offset * jnp.reshape(W_idct_hat, -1, order='F')

    return A1_hat, E1_hat, A_offset
