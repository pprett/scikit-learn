#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Principal Component Analysis
=========================================================

These figures aid in illustrating how a the point cloud
can be very flad in one direction - which is where PCA
would come in to choose a direction that is not flat.

"""
print __doc__

# Code source: Gael Varoqueux
# Modified for Documentation merge by Jaques Grobler
# License: BSD

import pylab as pl
import numpy as np
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

e = np.exp(1)
np.random.seed(4)


def pdf(x):
    return 0.5 * (stats.norm(scale=0.25 / e).pdf(x)
            + stats.norm(scale=4 / e).pdf(x))

y = np.random.normal(scale=0.5, size=(30000))
x = np.random.normal(scale=0.5, size=(30000))
z = np.random.normal(scale=0.1, size=len(x))

density = pdf(x) * pdf(y)
pdf_z = pdf(5 * z)

density *= pdf_z

a = x + y
b = 2 * y
c = a - b + z

norm = np.sqrt(a.var() + b.var())
a /= norm
b /= norm


###############################################################################
# Plot the figures
def plot_figs(fig_num, elev, azim):
    fig = pl.figure(fig_num, figsize=(4, 3))
    pl.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elev, azim=azim)

    ax.scatter(a[::10], b[::10], c[::10], c=density, marker='+',
            alpha=.4)
    Y = np.c_[a, b, c]
    U, pca_score, V = np.linalg.svd(Y, full_matrices=False)
    x_pca_axis, y_pca_axis, z_pca_axis = V.T * pca_score / pca_score.min()

    x_pca_axis, y_pca_axis, z_pca_axis = 3 * V.T
    x_pca_plane = np.r_[x_pca_axis[:2], - x_pca_axis[1::-1]]
    y_pca_plane = np.r_[y_pca_axis[:2], - y_pca_axis[1::-1]]
    z_pca_plane = np.r_[z_pca_axis[:2], - z_pca_axis[1::-1]]
    x_pca_plane.shape = (2, 2)
    y_pca_plane.shape = (2, 2)
    z_pca_plane.shape = (2, 2)
    ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])


elev = -40
azim = -80
plot_figs(1, elev, azim)

elev = 30
azim = 20
plot_figs(2, elev, azim)

pl.show()
