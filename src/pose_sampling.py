import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
device = torch.device("cuda:0")
import sys
from SPIN.models.smpl import SMPL


import numpy as np
from pytorch3d.renderer import (
    cameras,
    look_at_view_transform)
from UV_Volumes.lib.utils import data_utils

class Config():
    def __init__(self, ):
        self.voxel_size=[0.005, 0.005, 0.005]
        self.box_padding = 0.05
        self.ratio = 0.5
import torch.utils.data as data

class PoseSampler(data.Dataset):
    def __init__(self, 
                    device = torch.device("cuda:0"),
                    dtype=torch.float32,
                    pose_folder = './zju_mocap/CoreView_377/new_params', 
                    model_path = './SPIN/data/SMPL_NEUTRAL.pkl',
                    length = 500):
        
        super(PoseSampler, self).__init__()
        self.length = length
        self.poses, self.shapes = self.load_pose(pose_folder) #numpy
        pose_folder2="/home/yjli/AIGC/Adversarial_camou/zju_mocap/CoreView_377/new_params"
        poses2, shapes2 = self.load_pose(pose_folder2) #numpy
        if os.path.exists("./my_dataset"):
            pose_folder3="/home/yjli/AIGC/Adversarial_camou/my_dataset/Peter_chess/params"
            poses3, shapes3 = self.load_pose(pose_folder3) #numpy
            self.poses = np.concatenate((self.poses, poses2, poses3), axis=0)
        else:
            self.poses = np.concatenate((self.poses, poses2), axis=0)
        self.device = device
        self.dtype = dtype
        self.plot_pose_distribution(self.poses) 
        self.gmm = self.learn_pose_distribution_gmm()
        self.smpl = SMPL(model_path,
                    batch_size=1,
                    create_transl=False)
        self.smpl.eval()
        self.cfg = Config()
        self.batch_size = 1
            
    def pose2rot_batch(self, poses): # B, 72-> B, 24, 3, 3
        pose_rot = np.zeros((poses.shape[0], 24, 3,3))
        for k in range(poses.shape[0]):
            for i in range(24):
                pose_rot[k][i] = self.pose2rot(poses[k][i*3:(i+1)*3]) 
        return pose_rot

    def pose2rot(self, axis_angle):
        angle = np.linalg.norm(axis_angle)
        if angle == 0:
            return np.eye(3)
        axis = axis_angle / angle
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        cross_product_matrix = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = cos_angle * np.eye(3) + sin_angle * cross_product_matrix + (1 - cos_angle) * np.outer(axis, axis)
        return R

    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def plot_pose_distribution(self, pose_list, image_name = 'results/pose_distritution.png'):
        poses = pose_list  # Random data for demonstration

        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        poses_pca = pca.fit_transform(poses)

        # t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        poses_tsne = tsne.fit_transform(poses)

        # Plot PCA
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(poses_pca[:, 0], poses_pca[:, 1], alpha=0.5)
        plt.title('PCA of SMPL Poses')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Plot t-SNE
        plt.subplot(1, 2, 2)
        plt.scatter(poses_tsne[:, 0], poses_tsne[:, 1], alpha=0.5)
        plt.title('t-SNE of SMPL Poses')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.tight_layout()
        # Save the entire figure
        plt.savefig(image_name, format='png', dpi=300)  # Specify the path and file format

    def learn_pose_distribution_gmm(self,):
        from sklearn.mixture import GaussianMixture
        # Fit a Gaussian Mixture Model
        gmm = GaussianMixture(n_components=10, covariance_type='full')
        gmm.fit(self.poses)
        return gmm
        # Assuming you have a function to visualize or further process these samples
        # visualize_samples(samples)

    def load_pose(self, data_root="/home/yjli/AIGC/Adversarial_camou/zju_mocap/CoreView_377/new_params"):
        pose_list = []
        shape_list = []
        entries = os.listdir(data_root)
        for i in range(0, len(entries)):
            # transform smpl from the world coordinate to the smpl coordinate
            params_path = os.path.join(data_root, 
                                        '{}.npy'.format(i))
            params = np.load(params_path, allow_pickle=True).item()
            Rh = params['Rh']
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            Th = params['Th'].astype(np.float32)
            pose =  params["poses"]
            pose_list.append(pose[0])
            if "shapes"in params:
                shape =  params["shapes"]
                shape_list.append(shape[0])
            elif "beta" in params:
                shape =  params["beta"]
                shape_list.append(shape[0])
        pose_list = np.array(pose_list)
        shape_list = np.array(shape_list)
        # print("load poses with shape ", pose_list.shape)
        return pose_list, shape_list
    
    def sample_poses(self, ):
        samples = self.gmm.sample(self.batch_size)[0]
        return samples
    
    def sample_shapes(self, same =False):
        if same:
            shapes = np.expand_dims(self.shapes[0], 0).repeat(self.batch_size, axis=0)
        else:
            selected_indices = np.random.choice(shapes.shape[0], self.batch_size, replace=False)
            shapes = shapes[selected_indices, ...]
        return shapes

    def generate_vertices(self, pose_rot, shape_params):
        if pose_rot.shape[1]==72:
            pose_rot = self.pose2rot_batch(pose_rot) #numpy
        pose_rot = torch.tensor(pose_rot).to(self.dtype)
        shape_params = torch.tensor(shape_params).to(self.dtype)
        #pred_output = smpl(betas=shape_params, body_pose=pose_params, global_orient=pose_params, pose2rot=False)
        with torch.no_grad():
            pred_output = self.smpl(betas=shape_params, body_pose=pose_rot[:,1:], global_orient=pose_rot[:,0].unsqueeze(1), pose2rot=False)
        # Return the vertices
        return pred_output.vertices
    
    
    def camera_sampling(self, theta = 0, elev = 0):
        # thetas_list = np.linspace(-180, 180, angle_sample = self.batch)
        if theta is not None:
            if isinstance(theta, float) or isinstance(theta, int):
                self.azim = torch.zeros(self.batch_size).fill_(theta).float()
            elif isinstance(theta, torch.Tensor):
                self.azim = theta.clone().float()
            elif isinstance(theta, np.ndarray):
                self.azim = torch.from_numpy(theta).float()
            elif isinstance(theta, list):
                theta =np.array(theta)
                self.azim = torch.from_numpy(theta).float()
            else:
                raise ValueError
        else:
            if self.alpha > 0:
                exp = (self.alpha * self.sampler_probs).softmax(0)
                azim = torch.multinomial(exp, self.batch_size, replacement=True)
                self.azim_inds = azim
                azim = azim.to(exp)
                self.azim = (azim + azim.new(size=azim.shape).uniform_() - 0.5) * 360 / len(exp)
            else:
                self.azim_inds = None
                self.azim = (torch.zeros(self.batch_size).uniform_() - 0.5) * 360
        if elev is not None:
            elev = torch.zeros(self.batch_size).fill_(elev)
        else:
            elev = 10 + 8 * torch.zeros(self.batch_size).uniform_(-1, 1)
        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=self.azim)
        # Swap the second and third rows
        R_new = R
        R_new[:, 1], R_new[:, 2] = R[:, 2].clone(), R[:, 1].clone()
        return R_new, T
    
    def rotation_matrix_y(self, angle_degrees):
        """Generate a rotation matrix for a given angle in degrees around the y-axis."""
        angle_radians = np.radians(angle_degrees)
        c = np.cos(angle_radians)
        s = np.sin(angle_radians)
        return np.array([
            [c,  0, s],
            [0,  1, 0],
            [-s, 0, c]
        ])
    
    def prepare_input(self, xyz, random_rotation):
        Rh = self.rotation_matrix_y(random_rotation)
        Th = np.zeros(3)[None, :]

        xyz_rotated = xyz @ Rh.T + Th
        # PoseSampler.visualize_vertices(xyz,"results/right.png")

        min_xyz = np.min(xyz_rotated, axis=0)
        max_xyz = np.max(xyz_rotated, axis=0)
        min_xyz -= self.cfg.box_padding
        max_xyz += self.cfg.box_padding

        can_bounds = np.stack([min_xyz, max_xyz], axis=0)
        
        # R = cv2.Rodrigues(Rh)[0].astype(np.float32) # From Rotation Matrix to Vector or From  Vector to Rotation Matrix 
        # xyz = np.dot(xyz_rotated - Th, Rh) 
        # PoseSampler.visualize_vertices(xyz,"results/recover.png")

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= self.cfg.box_padding
        max_xyz += self.cfg.box_padding

        bounds = np.stack([min_xyz, max_xyz], axis=0)
        
        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(self.cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_shape = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_shape = (out_shape | (x - 1)) + 1 # make the output shape as the times of x
        
        # Rh = cv2.Rodrigues(Rh)
        return coord, Rh, Th, can_bounds, bounds, out_shape
        
    def __getitem__(self, index):
        self.batch_size = 1
        thetas_list = np.linspace(-180, 180, 50)
        theta_batch = np.array(np.random.choice(thetas_list, self.batch_size, replace=False))
        R, T = self.camera_sampling(theta_batch)
        R = R.detach().cpu().numpy()[0]
        T = T.detach().cpu().numpy().T 
        R_transform = np.array([[1,  0,   0],
                        [0,  0,  -1],
                        [0,  1,   0]], dtype=np.float32)

        R = np.dot(R_transform, R)

        # random_rotation = np.random.randint(-180, 180)
        # R = self.rotation_matrix_y(random_rotation)
        # T = np.zeros(3)[None, :].T

        # Example usage
        H , W = 512, 512
        fx, fy = 512, 512  # focal lengths in pixels
        cx, cy = 256, 256 
        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ])
        
        poses = self.sample_poses() # numpy
        shapes = self.sample_shapes(same=True) #numpy
        pose_rot = self.pose2rot_batch(poses) #numpy
        vertices = self.generate_vertices(pose_rot, shapes) # tensor
        xyz = vertices.detach().cpu().numpy()
        # random rotate the human body
        random_rotation = np.random.randint(-180, 180)
        coord,  Rh, Th, can_bounds, bounds, out_shape = self.prepare_input(xyz[0], random_rotation)
        img = np.ones((512, 512, 3))
        msk = np.ones((512, 512))
        iuv = np.ones((512, 512, 26))
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box, masked_iuv, mask_at_dp, mask_at_body, mask_at_bg, ray_d_center = \
                data_utils.sample_ray_h36m_whole(img, msk, K, R, T, can_bounds, iuv)

        ret = {
            'xyz':xyz, 
            'img_name': "000.png", 
            'coord': coord,
            'out_sh': out_shape,
            'rgb': rgb,
            'iuv': masked_iuv,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'mask_at_dp': mask_at_dp,
            'mask_at_body': mask_at_body,
            'mask_at_bg': mask_at_bg,
            'poses': poses[0],
            'ratio': np.array(self.cfg.ratio, dtype=np.float32),
            'ray_d_center': ray_d_center.astype(np.float32),
            'bounds': bounds,
            'R': Rh,
            'Th': Th,
            'latent_index': 0,
            'frame_index': 0,
            'cam_ind': 0,
            'H': H,
            'W': W,
        }
        return ret


    def Histogram(pose_parameters, imagename="./results/poses_histogram.png"):
        # Set up the figure and axes for plotting
        fig, axes = plt.subplots(nrows=12, ncols=6, figsize=(30, 40))  # Adjust the size as necessary
        axes = axes.flatten()  # Flatten the 2D array of axes objects for easy iteration

        # Loop through each parameter index
        for i in tqdm(range(72)):
            # Collect all values for the i-th parameter across all poses
            param_values = [pose[i] for pose in pose_parameters]
            
            # Plot histogram for the i-th parameter
            ax = axes[i]
            ax.hist(param_values, bins=20, color='blue', alpha=0.7)
            ax.set_title(f'Parameter {i+1}')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        #plt.show()
        plt.savefig(imagename, format='png', dpi=300)  # Specify the path and file format

    def visualize_vertices(data, imagename="results/vertices.png"):
        """
        Visualizes the vertices in a 3D plot.
        
        Args:
        vertices (np.array): Vertices to plot.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 12))  # Adjust the size as necessary
        axes = axes.flatten()  # Flatten the 2D array of axes objects for easy iteration
        for i in range(data.shape[0]):
            # Extract the point cloud data for the i-th figure
            x = data[i, :, 0]
            y = data[i, :, 1]
            z = data[i, :, 2]
            ax = axes[i]
            point_size = 0.5
            ax.scatter(x, y,  c='r', marker='o',s = point_size)  # Plot the points
            ax.set_title(f'3D Point Cloud {i+1}')
        plt.tight_layout()
        #plt.show()
        plt.savefig(imagename, format='png', dpi=300)  # Specify the path and file format
    # Example usage

    def visualize_vertices_3D(data, imagename="results/vertices3D.png"):
        """
        Visualizes the vertices in a 3D plot.
        
        Args:
        vertices (np.array): Vertices to plot.
        """
        fig = plt.figure()
        for i in range(data.shape[0]):
            ax = fig.add_subplot(4,5,i+1, projection='3d')

            # Extract the point cloud data for the i-th figure
            x = data[i, :, 0]
            y = data[i, :, 1]
            z = data[i, :, 2]
            point_size = 0.2
            ax.invert_yaxis()
            ax.scatter(x,z,y, c='r', marker='o', s = point_size)  # Plot the points
            ax.set_axis_off()
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.set_title(f'3D Point Cloud {i+1}')
        plt.tight_layout()
        #plt.show()
        plt.savefig(imagename, format='png', dpi=300)  # Specify the path and file format
        
    def visualize_GMM(self, imagename="results/pca-gmm.png"):
        from sklearn.decomposition import PCA
        from scipy.stats import multivariate_normal
        # Apply PCA to reduce the dimensionality to 2D
        pca = PCA(n_components=2)
        poses_reduced = pca.fit_transform(self.poses)

        # Estimate the parameters of the Gaussian distribution from the reduced data
        mean = np.mean(poses_reduced, axis=0)
        covariance = np.cov(poses_reduced, rowvar=False)

        # Create a grid of points for plotting the Gaussian distribution
        x, y = np.meshgrid(np.linspace(np.min(poses_reduced[:, 0]), np.max(poses_reduced[:, 0]), 100),
                        np.linspace(np.min(poses_reduced[:, 1]), np.max(poses_reduced[:, 1]), 100))
        pos = np.dstack((x, y))

        # Evaluate the Gaussian distribution at the grid points
        rv = multivariate_normal(mean, covariance)
        z = rv.pdf(pos)

        # Plot the reduced poses and the Gaussian distribution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)
        # ax.scatter(poses_reduced[:, 0], poses_reduced[:, 1], c='red', s=10)
        ax.set_axis_off()
        ax.view_init(elev=30, azim=0)
        # ax.set_xlabel('PCA Component 1')
        # ax.set_ylabel('PCA Component 2')
        # ax.set_zlabel('Probability Density')
        plt.savefig(imagename, format='png', dpi=500) 
        plt.show()

    
    def __len__(self):
        return self.length
    
# plot_pose_distribution(samples,"results/new_pose_samples.png")
# Histogram(pose_list,"results/gt_pose_Histogram.png")
# Histogram(samples, "results/generated_pose_Histogram.png")






if __name__ == "__main__":
    model_path = '/home/yjli/AIGC/Adversarial_camou/SPIN/data/SMPL_NEUTRAL.pkl'
    # model_path = "/home/yjli/AIGC/humannerf/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    pose_sampler = PoseSampler()
    poses = pose_sampler.sample_poses() # numpy
    shapes = pose_sampler.sample_shapes(same=True) #numpy
    pose_sampler.visualize_GMM()
    np.savetxt('poses.txt', pose_sampler.poses)

    pose_sampler.__getitem__(0)
    # pose_rot = pose_sampler.pose2rot_batch(poses) #numpy
    # vertices = pose_sampler.generate_vertices(pose_rot, shapes) # tensor
    # vertices = vertices.detach().cpu().numpy()
    # PoseSampler.visualize_vertices_3D(vertices)

    