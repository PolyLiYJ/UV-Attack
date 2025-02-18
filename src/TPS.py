#https://github.com/cheind/py-thin-plate-spline/blob/master/TPS.ipynb

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import torch
import torch.nn.functional as F


import numpy as np

class TPS:       
    @staticmethod
    def fit(c, lambd=0., reduced=False):        
        n = c.shape[0]

        U = TPS.u(TPS.d(c, c))
        K = U + np.eye(n, dtype=np.float32)*lambd

        P = np.ones((n, 3), dtype=np.float32)
        P[:, 1:] = c[:, :2]

        v = np.zeros(n+3, dtype=np.float32)
        v[:n] = c[:, -1]

        A = np.zeros((n+3, n+3), dtype=np.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        theta = np.linalg.solve(A, v) # p has structure w,a
        return theta[1:] if reduced else theta
        
    @staticmethod
    def d(a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r**2 * np.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        x = np.atleast_2d(x)
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = np.concatenate((-np.sum(w, keepdims=True), w))
        b = np.dot(U, w)
        return a[0] + a[1]*x[:, 0] + a[2]*x[:, 1] + b
    

def uniform_grid(shape):
    '''Uniform grid coordinates.
    
    Params
    ------
    shape : tuple
        HxW defining the number of height and width dimension of the grid

    Returns
    -------
    points: HxWx2 tensor
        Grid coordinates over [0,1] normalized image range.
    '''

    H,W = shape[:2]    
    c = np.empty((H, W, 2))
    c[..., 0] = np.linspace(0, 1, W, dtype=np.float32)
    c[..., 1] = np.expand_dims(np.linspace(0, 1, H, dtype=np.float32), -1)

    return c
    
def tps_theta_from_points(c_src, c_dst, reduced=False):
    delta = c_src - c_dst
    
    cx = np.column_stack((c_dst, delta[:, 0]))
    cy = np.column_stack((c_dst, delta[:, 1]))
        
    theta_dx = TPS.fit(cx, reduced=reduced)
    theta_dy = TPS.fit(cy, reduced=reduced)

    return np.stack((theta_dx, theta_dy), -1)

def tps_grid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.
    
    Params
    ------
    grid : HxWx2 array
        Normalized flow field coordinates as computed by compute_densegrid.
    sshape : tuple
        Height and width of source image in pixels.


    Returns
    -------
    mapx : HxW array
    mapy : HxW array
    '''

    mx = (grid[:, :, 0] * sshape[1]).astype(np.float32)
    my = (grid[:, :, 1] * sshape[0]).astype(np.float32)

    return mx, my



def tps(theta, ctrl, grid):
    '''Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
    The TPS surface is a minimum bend interpolation surface defined by a set of control points.
    The function value for a x,y location is given by
    
        TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])
        
    This method computes the TPS value for multiple batches over multiple grid locations for 2 
    surfaces in one go.
    
    Params
    ------
    theta: Nx(T+3)x2 tensor, or Nx(T+2)x2 tensor
        Batch size N, T+3 or T+2 (reduced form) model parameters for T control points in dx and dy.
    ctrl: NxTx2 tensor or Tx2 tensor
        T control points in normalized image coordinates [0..1]
    grid: NxHxWx3 tensor
        Grid locations to evaluate with homogeneous 1 in first coordinate.
        
    Returns
    -------
    z: NxHxWx2 tensor
        Function values at each grid location in dx and dy.
    '''
    
    N, H, W, _ = grid.size()

    if ctrl.dim() == 2:
        ctrl = ctrl.expand(N, *ctrl.size())
    
    T = ctrl.shape[1]
    
    diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)
    D = torch.sqrt((diff**2).sum(-1))
    U = ((D**2) * torch.log(D + 1e-6)).float()

    w, a = theta[:, :-3, :], theta[:, -3:, :]

    reduced = T + 2  == theta.shape[1]
    if reduced:
        w = torch.cat((-w.sum(dim=1, keepdim=True), w), dim=1) 

    # U is NxHxWxT
    b = torch.bmm(U.view(N, -1, T), w).view(N,H,W,2)
    # b is NxHxWx2
    z = torch.bmm(grid.view(N,-1,3), a).view(N,H,W,2) + b
    
    return z

def tps_grid(theta, ctrl, size):
    '''Compute a thin-plate-spline grid from parameters for sampling.
    
    Params
    ------
    theta: Nx(T+3)x2 tensor
        Batch size N, T+3 model parameters for T control points in dx and dy.
    ctrl: NxTx2 tensor, or Tx2 tensor
        T control points in normalized image coordinates [0..1]
    size: tuple
        Output grid size as NxCxHxW. C unused. This defines the output image
        size when sampling.
    
    Returns
    -------
    grid : NxHxWx2 tensor
        Grid suitable for sampling in pytorch containing source image
        locations for each output pixel.
    '''    
    N, _, H, W = size

    grid = theta.new(N, H, W, 3)
    grid[:, :, :, 0] = 1.
    grid[:, :, :, 1] = torch.linspace(0, 1, W)
    grid[:, :, :, 2] = torch.linspace(0, 1, H).unsqueeze(-1)   
    
    z = tps(theta, ctrl, grid)
    return (grid[...,1:] + z)*2-1 # [-1,1] range required by F.sample_grid

def tps_sparse(theta, ctrl, xy):
    if xy.dim() == 2:
        xy = xy.expand(theta.shape[0], *xy.size())

    N, M = xy.shape[:2]
    grid = xy.new(N, M, 3)
    grid[..., 0] = 1.
    grid[..., 1:] = xy

    z = tps(theta, ctrl, grid.view(N,M,1,3))
    return xy + z.view(N, M, 2)

def uniform_grid(shape):
    '''Uniformly places control points aranged in grid accross normalized image coordinates.
    
    Params
    ------
    shape : tuple
        HxW defining the number of control points in height and width dimension

    Returns
    -------
    points: HxWx2 tensor
        Control points over [0,1] normalized image range.
    '''
    H,W = shape[:2]    
    c = torch.zeros(H, W, 2)
    c[..., 0] = torch.linspace(0, 1, W)
    c[..., 1] = torch.linspace(0, 1, H).unsqueeze(-1)
    return c

def show_warped(img, warped,  c_src, c_dst):
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(img[...,::-1], origin='upper')
    axs[0].scatter(c_src[:, 0]*img.shape[1], c_src[:, 1]*img.shape[0], marker='+', color='black')
    axs[1].imshow(warped[...,::-1], origin='upper')
    axs[1].scatter(c_dst[:, 0]*warped.shape[1], c_dst[:, 1]*warped.shape[0], marker='+', color='black')
    plt.savefig("wrappedimg.png")
    
def warp_image_cv(img, c_src, c_dst, theta, dshape=None):
    dshape = dshape or img.shape
    grid = tps_grid(theta, c_dst, dshape)
    # mapx, mapy = tps_grid_to_remap(grid, img.shape)
    output = F.grid_sample(img, grid)
    # return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
    return output

import cv2
img = cv2.imread('/home/yjli/AIGC/Adversarial_camou/results/experiment/256x256/output.jpg')
# img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb  = np.array(img)
input = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0)
print(img.shape)
c_src = np.array([
    [0.0, 0.0],
    [1., 0],
    [1, 1],
    [0, 1],
    [0.3, 0.3],
    [0.7, 0.7],
])

image = cv2.imread('/home/yjli/AIGC/Adversarial_camou/my_dataset/A1/00716.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
num_cols, num_rows = 7, 11      
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
ret, corners = cv2.findChessboardCorners(gray, (num_cols, num_rows), flags)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

if ret == True:
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    cv2.drawChessboardCorners(image, (num_cols, num_rows), corners, ret)
    cv2.imwrite('Chessboard.png', image)

c_dst = np.array([
    [0., 0],
    [1., 0],    
    [1, 1],
    [0, 1],
    [0.4, 0.4],
    [0.6, 0.6],
])


theta = tps_theta_from_points(c_src, c_dst, reduced=True) # compute TPS coefficients, using numpy

theta = torch.from_numpy(theta).unsqueeze(0)
c_src_tensor = torch.from_numpy(c_src).unsqueeze(0)
c_dst_tensor = torch.from_numpy(c_dst).unsqueeze(0)
wraped = warp_image_cv(input, c_src_tensor, c_dst_tensor, theta, dshape=(1, 3, 512, 512))
wraped = ((wraped[0].permute(1,2,0).cpu().detach().numpy()))
show_warped(img_rgb, wraped, c_src, c_dst)