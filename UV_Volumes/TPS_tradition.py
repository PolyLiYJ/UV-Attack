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
import matplotlib.pyplot as plt
from PIL import Image

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
    # U = ((D**2) * torch.log(D + 1e-6)).float()
    U = ((D**2) * torch.log(D + 1e-6)).to(theta.dtype)

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

def show_warped(img, warped, target, c_src, c_dst, imgname = "wrappedimg.png"):
    fig, axs = plt.subplots(1, 3, figsize=(24,8))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    axs[0].imshow(img, origin='upper')
    #axs[0].scatter(c_src[:, 0]*img.shape[1], c_src[:, 1]*img.shape[0], marker='+', color='red')
    axs[1].imshow(warped, origin='upper')
    #axs[1].scatter(c_dst[:, 0]*warped.shape[1], c_dst[:, 1]*warped.shape[0], marker='+', color='red')
    axs[2].imshow(target, origin='upper')
    #axs[2].scatter(c_dst[:, 0]*target.shape[1], c_dst[:, 1]*target.shape[0], marker='+', color='red')
    plt.savefig(imgname)
    
def warp_image_cv(img, c_src, c_dst, theta, dshape=None):
    dshape = dshape or img.shape
    grid = tps_grid(theta, c_dst, dshape)
    # mapx, mapy = tps_grid_to_remap(grid, img.shape)
    output = F.grid_sample(img, grid, mode='bilinear', align_corners=False)
    return output, grid

if __name__ == "__main__":
    # Open the image using PIL
    image_pil = Image.open('/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth2.png') # range 0,1

    # Convert image to RGB
    image_rgb = image_pil.convert('RGB')

    # Save the converted image
    image_rgb.save('/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth2.png')

    import torchvision.transforms as transforms
    img = cv2.imread('/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth.png')
    img_size = 256
    shape = (1,3,img_size, img_size)
    source = np.array(img)/255
    input = torch.tensor(source).permute(2,0,1).float().unsqueeze(0)

    """image = cv2.imread('/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_cols, num_rows = 7, 11      
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, (num_cols, num_rows), flags)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if ret == True:
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(image, (num_cols, num_rows), corners, ret)
        cv2.imwrite('Chessboard.png', image)
    """

    chess_coordinate = np.array([
        [0,0],
        [0,13],
        [17,0],
        [17,13],
        [8, 4],
        [13, 9],
        [14, 5],
        [5, 8]
        ])
        
    # c_src = np.array([
    #     [129.5, 164],
    #     [129.5, 406],
    #     [446.5, 163],
    #     [445.5, 406],
    #     [276.5, 239],
    #     [371.5, 331],
    #     [394.5, 254],
    #     [222.5, 314]
    # ])

    c_src = np.array([[0.73818,0.23545],
    [0.29455,0.23545],
    [0.73818,0.81182],
    [0.29455,0.81364],
    [0.74182,0.33545],
    [0.64182,0.34455],
    [0.53636,0.33545],
    [0.43818,0.33364],
    [0.29818,0.33182],
    [0.73818,0.44091],
    [0.64000,0.44091],
    [0.53273,0.44091],
    [0.43091,0.43727],
    [0.29455,0.43727],
    [0.73818,0.54273],
    [0.63636,0.53909],
    [0.53273,0.54273],
    [0.43455,0.54273],
    [0.29636,0.54273],
    [0.74182,0.64636],
    [0.63455,0.64273],
    [0.53636,0.64636],
    [0.43455,0.64636],
    [0.29818,0.64273],
    [0.74364,0.74818],
    [0.63455,0.74818],
    [0.53273,0.74273],
    [0.43091,0.74818],
    [0.29818,0.74273],
    [0.63636,0.81364],
    [0.53455,0.81364],
    [0.42909,0.81364]])

    # c_dst = np.array([
    #     [61, 72],
    #     [57, 186],    
    #     [177, 61],
    #     [185, 189],
    #     [107, 100],
    #     [142, 147],
    #     [146, 111],
    #     [98.38, 132.84]
    # ])

    c_dst = np.array([[0.72163,0.22735],
    [0.28456,0.24337],
    [0.73536,0.73079],
    [0.24566,0.70561],
    [0.71248,0.35092],
    [0.58204,0.34635],
    [0.50195,0.34406],
    [0.41042,0.31202],
    [0.29600,0.33491],
    [0.70561,0.42415],
    [0.57747,0.42186],
    [0.47678,0.39669],
    [0.39898,0.38754],
    [0.27770,0.42644],
    [0.69646,0.49509],
    [0.58891,0.49051],
    [0.46763,0.46534],
    [0.39898,0.44246],
    [0.26397,0.53170],
    [0.71477,0.57976],
    [0.59806,0.55230],
    [0.49051,0.51568],
    [0.38754,0.51339],
    [0.26626,0.58662],
    [0.72163,0.65756],
    [0.62095,0.63010],
    [0.51339,0.60035],
    [0.37610,0.60950],
    [0.25481,0.63468],
    [0.62323,0.72163],
    [0.49966,0.71019],
    [0.36694,0.69646]])

    theta = tps_theta_from_points(c_src, c_dst, reduced=True) # compute TPS coefficients, using numpy

    theta_tensor = torch.from_numpy(theta).unsqueeze(0)
    c_src_tensor = torch.from_numpy(c_src).unsqueeze(0)
    c_dst_tensor = torch.from_numpy(c_dst).unsqueeze(0)
    wraped, grid = warp_image_cv(input, c_src_tensor, c_dst_tensor, theta_tensor, dshape=shape)
    wraped = ((wraped[0].permute(1,2,0).clamp(0,1).cpu().detach().numpy()))
    
    deformed_grid = torch.zeros((1, 256, 256, 3))
    deformed_grid[:, :, :, 0] = 0
    deformed_grid[:, :, :, 1:3] = (grid+1)/2
    deformed_grid= ((deformed_grid[0].clamp(0,1).cpu().detach().numpy()))
    
    ori_grid = torch.zeros((1, 256, 256, 3))
    ori_grid[:, :, :, 0] = 0
    ori_grid[:, :, :, 1] = torch.linspace(0, 1, 256)
    ori_grid[:, :, :, 2] = torch.linspace(0, 1, 256).unsqueeze(-1)  
    ori_grid= ((ori_grid[0].clamp(0,1).cpu().detach().numpy()))

    img = cv2.imread('/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/tps_dataset/images/00013_01.png')
    # img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target  = np.array(img)
    show_warped(source, wraped, target, c_src, c_dst)
    show_warped(ori_grid, deformed_grid, deformed_grid, c_src, c_dst,"grid.png")

    cv2.imwrite("wraped_front.png", wraped * 255)

    tps_para_front = {}
    tps_para_front["theta_tensor"] = theta_tensor
    tps_para_front["c_src_tensor"] = c_src_tensor
    tps_para_front["c_dst_tensor"] = c_dst_tensor
    torch.save( tps_para_front, "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/trained_model/TPS/tps_para_front.pt")

    import torchvision.transforms as transforms
    img = cv2.imread('/home/yjli/AIGC/Adversarial_camou/UV_Volumes/cloth2.png')
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape[0]
    print(img_size)
    shape = (1,3,img_size, img_size)
    source = np.array(img)/255
    input = torch.tensor(source).permute(2,0,1).float().unsqueeze(0)

    chess_coordinate = np.array([
        [0,0],
        [0,13],
        [17,0],
        [17,13],
        [8, 4],
        [13, 9],
        [14, 5],
        [5, 8]
        ])
        
    c_src = np.array([[0.28545,0.26455],
                    [0.43091,0.26455],
                    [0.56727,0.26455],
                    [0.71455,0.26273],
                    [0.28727,0.40636],
                    [0.42545,0.40455],
                    [0.57091,0.40455],
                    [0.72,0.40455],
                    [0.28545,0.54455],
                    [0.42545,0.54273],
                    [0.56727,0.54455],
                    [0.71273,0.54273],
                    [0.28545,0.69],
                    [0.41636,0.69182],
                    [0.56909,0.69545],
                    [0.72,0.69545]]
    )


    c_dst= np.array(
    [[0.16016,0.24609],
    [0.35547,0.23828],
    [0.51953,0.24219],
    [0.69922,0.23828],
    [0.17969,0.41797],
    [0.35547,0.40625],
    [0.51172,0.40625],
    [0.70313,0.40625],
    [0.17188,0.5625],
    [0.36719,0.55469],
    [0.52734,0.55469],
    [0.71484,0.55859],
    [0.17578,0.71094],
    [0.33594,0.71484],
    [0.53906,0.71875],
    [0.69922,0.72656]])

    theta = tps_theta_from_points(c_src, c_dst, reduced=True) # compute TPS coefficients, using numpy

    theta_tensor = torch.from_numpy(theta).unsqueeze(0)
    c_src_tensor = torch.from_numpy(c_src).unsqueeze(0)
    c_dst_tensor = torch.from_numpy(c_dst).unsqueeze(0)
    wraped, grid = warp_image_cv(input, c_src_tensor, c_dst_tensor, theta_tensor, dshape=shape)
    wraped = ((wraped[0].permute(1,2,0).clamp(0,1).cpu().detach().numpy()))

    img = cv2.imread('/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/tps_dataset/images/00013_00.png')
    # img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target  = np.array(img)
    show_warped(source, wraped, target, c_src, c_dst, "wraped_image_back.png")
    cv2.imwrite("wraped_back.png", wraped * 255)

    tps_para_front = {}
    tps_para_front["theta_tensor"] = theta_tensor
    tps_para_front["c_src_tensor"] = c_src
    tps_para_front["c_dst_tensor"] = c_dst
    torch.save( tps_para_front, "/home/yjli/AIGC/Adversarial_camou/UV_Volumes/data/trained_model/TPS/tps_para_back.pt")