import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops import knn_points
from plyfile import PlyData, PlyElement
import time
import argparse

# Load data from ply file
def load_from_ply(path: str, sh_degrees: int = -1):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(sh_degrees + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_degrees + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    return xyz,features_dc,features_extra,opacities,scales,rots


# Save to ply file
def save_to_ply(path,gaussians):
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    xyz = gaussians[0]
    normals = gaussians[1]
    opacities = gaussians[2]
    f_dc = torch.tensor(gaussians[3]).flatten(start_dim=1).contiguous().cpu().numpy() #N*3*1
    f_rest = torch.tensor(gaussians[4]).flatten(start_dim=1).contiguous().cpu().numpy() # N*3*15

    scale = gaussians[5]
    rotation = gaussians[6]


    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

# Load implicit function grid data from a binary file
def load_sampled_implicit_function(file_path):
    with open(file_path, 'rb') as f:
        # Read the ASCII header lines
        header = []
        for _ in range(7):
            header.append(f.readline().decode('ascii').strip())

        # Extract information from the header
        resolution = list(map(int, header[2].split()))
        matrix = np.array([list(map(float, header[i].split())) for i in range(3, 7)])
        # Calculate the total number of grid values
        total_values = np.prod(resolution)
        
        # Read the binary data (float32 values)
        grid_values = np.fromfile(f, dtype=np.float32, count=total_values)
        # Reshape the grid to match the resolution (RES_X, RES_Y, RES_Z)
        grid = grid_values.reshape(resolution)
    return grid, matrix,resolution[0]

# Extract points near-zero from grid
def extract_near_zero_points(grid, transformation_matrix, threshold=0.5):
    # Obtain the resolution of the grid
    res_x, res_y, res_z = grid.shape
    # Find the index of the grid value that is close to zero
    near_zero_indices = np.argwhere(grid > threshold)
    #near_zero_indices = np.argwhere(abs(grid-0.5) < threshold)
    # Optionally, you can extract the corresponding SDF values at those indices:
    # 
    near_zero_values = grid[tuple(near_zero_indices.T)] 

    # Swap the first and last columns of near_zero_indices
    near_zero_indices[:, [0, 2]] = near_zero_indices[:, [2, 0]]

    # Adjust the index to conform to the half-integer (dual grid) representation method, with grid coordinates + 0.5
    half_shifted_coords = (near_zero_indices + 0.5)
    
    # Convert grid coordinates to homogeneous coordinates [x, y, z, 1]
    homogeneous_coords = np.ones((half_shifted_coords.shape[0], 4))
    homogeneous_coords[:, :3] = half_shifted_coords

    # Apply the transformation matrix to convert to world coordinates
    # Note: If the transformation matrix is not orthogonal, you may need to transpose the rotation part
    # If the transformation matrix is orthogonal, you can directly apply it 
    #transformation_matrix[:3,:3] = transformation_matrix[:3,:3].T
    world_coords_homogeneous = (transformation_matrix @ homogeneous_coords.T).T

    # Do not perform normalization, directly extract the 3D coordinates (x, y, z)
    world_coords = world_coords_homogeneous[:, :3]

    return world_coords,near_zero_values

# Save pointcloud using open3d
def save_point_cloud(points, file_name):
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    
    # Assign the points to the point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Save the point cloud to a .ply file
    o3d.io.write_point_cloud(file_name, pcd)

    print(f"Point cloud saved to {file_name}")

# Convert world coordinates to original scene grid coordinates
def transform_to_original_scene_coordinates(world_coords, transformation_matrix):
    # Convert world coordinates to homogeneous coordinates [x, y, z, 1]
    homogeneous_world_coords = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))

    # Compute the inverse of the transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Apply the inverse transformation
    original_coords_homogeneous = (inverse_transformation_matrix @ homogeneous_world_coords.T).T
    
    # Extract the original coordinates (x, y, z) from the homogeneous coordinates
    original_coords = original_coords_homogeneous[:, :3] - 0.5
    original_coords[:, [0, 2]] = original_coords[:, [2, 0]]
    return original_coords

# Create a mask from target points
def create_mask_from_target_pts(target_pts_on_obj, scene_sdf,scene_res,values_near_zero):
    # Initialize mask with zeros
    mask = np.ones(target_pts_on_obj.shape[0], dtype=bool)

    # Extract x, y, z coordinates from target points
    x, y, z = target_pts_on_obj[:, 0], target_pts_on_obj[:, 1], target_pts_on_obj[:, 2]

    # Create an array of offsets for the neighbors
    offsets = np.array([
        [-1, -1, -1], [-1, -1, 1],
        [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1],
        [1, 1, -1], [1, 1, 1]
    ])

    # Calculate all neighbor coordinates
    neighbor_coords = np.array([x, y, z]).T[:, np.newaxis, :] + offsets  # Shape: (N, 8, 3)

    # Clip the neighbor coordinates to be within valid bounds [0, 255]
    neighbor_coords = np.clip(neighbor_coords, 0, int(scene_res-1))

    # Check for values in scene_sdf that are < 0.5 for all neighbors
    # We need to gather the values from the scene_sdf for the neighbors
    neighbor_indices = neighbor_coords.reshape(-1, 3)  # Reshape to (N*8, 3)
    sdf_values = scene_sdf[neighbor_indices[:, 0], neighbor_indices[:, 1], neighbor_indices[:, 2]]  # Gather sdf values

    # Reshape the sdf values back to (N, 8)
    sdf_values = sdf_values.reshape(target_pts_on_obj.shape[0], 8)
    
    # Create a mask where any value in the neighbors is < 0.5

    mask[np.sum(sdf_values > 0.5, axis=1) >=2] = False
    values_near_zero[np.sum(sdf_values > 0.5, axis=1) >=1] = 0.501
    return mask,values_near_zero


def otherInterpolation(tgt_pts, src_pts, src_fea_dc):
    # Define an interpolation function to perform weighted averaging for arbitrary features
    def interpolate_features(features, idx_nearest, weights):
        nearest_features = features[idx_nearest]  # Retrieve the features of the nearest neighbors, shape = (M, K, F)
        interpolated_features = (nearest_features * weights.unsqueeze(-1)).sum(dim=1)  # shape = (M, F)
        return interpolated_features

    K = 100  # Number of nearest neighbors
    knn_result = knn_points(
        tgt_pts.unsqueeze(0).to(torch.float32).cuda(),
        src_pts.unsqueeze(0).to(torch.float32).cuda(),
        K=K
    )

    idx_nearest = knn_result.idx.squeeze(0)   # Indices of the nearest neighbors, shape = (M, K)
    dists_nearest = knn_result.dists.squeeze(0)  # Distances to the nearest neighbors, shape = (M, K)
    print('dists_nearest: ', dists_nearest.shape)

    # Step 2: Use Inverse Distance Weighting (IDW) via a Gaussian kernel to compute weights
    sigma = 50.0
    # weights = 1.0 / (dists_nearest + 1e-8)  # Alternative: classic IDW (commented to use Gaussian instead)
    weights = torch.exp(-dists_nearest / (2 * sigma ** 2))  # Gaussian-based weights
    weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights, shape = (M, K)

    # Perform weighted interpolation of the source features
    add_surface_points_features_dc = interpolate_features(
        src_fea_dc.reshape(len(src_fea_dc), -1), 
        idx_nearest, 
        weights
    )[:, None, :]  # Add an extra dimension for consistency

    return add_surface_points_features_dc

def interpolate_gaussians(
    source_pts, 
    source_colors, 
    dense_points=None, 
    dense_radius=None, 
    dense_values=None, 
    obj_raw_gaussian=None
):
    def inverse_sigmoid(x):
        return torch.log(x / (1 - x))

    def RGB2SH(rgb):
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0
    
    # Interpolation helper using weighted sum
    def interpolate_features(features, idx_nearest, weights):
        nearest_features = features[idx_nearest]  # (M, K, F)
        interpolated_features = (nearest_features * weights.unsqueeze(-1)).sum(dim=1)  # (M, F)
        return interpolated_features
    
    device = torch.device("cuda")
    
    K = 100  # K nearest neighbors
    dense_points_tensor = torch.tensor(dense_points, dtype=torch.float32, device=device)
    source_pts = source_pts.to(device).float()
    source_colors = source_colors.to(device).float()
    source_shs = RGB2SH(source_colors)

    # KNN Computation
    print("Computing KNN...")
    t1 = time.time()
    knn_result = knn_points(dense_points_tensor.unsqueeze(0), source_pts.unsqueeze(0), K=K)
    t2 = time.time()
    print('KNN computation time:', t2 - t1)
    
    idx_nearest = knn_result.idx.squeeze(0)[:, 0]  # nearest index (M,)
    nearest_distances = knn_result.dists.squeeze(0)[:, 0]  # nearest distances (M,)
    
    # Compute interpolation weights
    sigma = 10.0
    weights = torch.exp(-knn_result.dists.squeeze(0) / (2 * sigma ** 2))
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    # Masking points too close to original points
    threshold_distance = 0.01 * nearest_distances.mean()
    nearst_point_mask = nearest_distances >= threshold_distance
    
    # Initialize dense features
    dense_features_dc = torch.zeros((*dense_points_tensor.shape, 1), device=device)
    interpolated_shs = interpolate_features(source_shs, knn_result.idx.squeeze(0), weights)
    dense_features_dc[nearst_point_mask] = interpolated_shs[nearst_point_mask].unsqueeze(-1)
    
    dense_opacities = inverse_sigmoid(0.5 * torch.ones(len(dense_points_tensor), 1, device=device))
    dense_normals = torch.zeros_like(dense_points_tensor, device=device)
    dense_scales = torch.log(0.7 * dense_radius * torch.ones_like(dense_points_tensor, device=device))
    dense_rots = torch.zeros((len(dense_opacities), 4), device=device)
    dense_rots[:, 0] = 1.0
    dense_features_extra = torch.zeros((len(dense_points_tensor), 3, 15), device=device)
    
    # Original point cloud definition
    xyz = source_pts
    normals = torch.zeros_like(xyz, device=device)
    features_dc = source_shs.unsqueeze(-1)
    features_extra = torch.zeros((len(xyz), 3, 15), device=device)
    opacities = inverse_sigmoid(0.9 * torch.ones(len(xyz), 1, device=device))
    rots = torch.zeros((len(xyz), 4), device=device)
    rots[:,0] = 1.0
    scales = torch.log(0.7 * dense_radius * torch.ones_like(xyz, device=device))
    
    # Determine inlier mask for the original point cloud
    print("Calculating inlier mask...")
    t1 = time.time()
    knn_result_new = knn_points(xyz.unsqueeze(0), dense_points_tensor.unsqueeze(0), K=1)
    distances_new = knn_result_new.dists.squeeze(0)[:, 0]
    inlier_threshold = 1.5 * distances_new.mean()
    inlier_mask = distances_new < inlier_threshold
    t2 = time.time()
    print('Inlier mask computation time:', t2 - t1)
    
    # Combine inliers and interpolated points
    dense_points_final = torch.cat((xyz[inlier_mask], dense_points_tensor[nearst_point_mask]), dim=0).cpu()
    dense_normals_final = torch.cat((normals[inlier_mask], dense_normals[nearst_point_mask]), dim=0).cpu()
    dense_opacities_final = torch.cat((opacities[inlier_mask], dense_opacities[nearst_point_mask]), dim=0).cpu()
    dense_features_extra_final = torch.cat((features_extra[inlier_mask], dense_features_extra[nearst_point_mask]), dim=0).cpu()
    dense_features_dc_final = torch.cat((features_dc[inlier_mask], dense_features_dc[nearst_point_mask]), dim=0).cpu()
    dense_scales_final = torch.cat((scales[inlier_mask], dense_scales[nearst_point_mask]), dim=0).cpu()
    dense_rots_final = torch.cat((rots[inlier_mask], dense_rots[nearst_point_mask]), dim=0).cpu()

    return (
        dense_points_final,
        dense_normals_final,
        dense_opacities_final,
        dense_features_dc_final,
        dense_features_extra_final,
        dense_scales_final,
        dense_rots_final
    )

def load_pointcloud(path):
    # Load the point cloud from a .ply file
    point_cloud = o3d.io.read_point_cloud(path)
    # Access the points (as a numpy array)
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    return torch.tensor(points).to(torch.float32),torch.tensor(colors).to(torch.float32)

if __name__ == "__main__":
    # Example usage load grids to solve conflicits
    parser = argparse.ArgumentParser()
    parser.add_argument("object_name", type=str, help="object name, e.g., bear")
    args = parser.parse_args()
    this_name = args.object_name
    save_path = "../exp_res/"+this_name+"/dense_poisson_scripts/"
    
    obj_path = save_path+this_name+"_results/"+'obj_mesh_0.bin'
    obj_grid, obj_transformation_matrix,obj_res = load_sampled_implicit_function(obj_path)
    obj_gaus_radius = obj_transformation_matrix[0,0]
    
    scene_path =save_path + this_name+"_results/"+'scene_mesh.bin'
    scene_grid, scene_transformation_matrix,scene_res = load_sampled_implicit_function(scene_path)

    # Extract points near zero in grid and transform them to world coordinates
    points_near_zero,values_near_zero = extract_near_zero_points(obj_grid, obj_transformation_matrix)

    target_pts_on_obj= transform_to_original_scene_coordinates(points_near_zero, scene_transformation_matrix)
    mask,values_near_zero = create_mask_from_target_pts(target_pts_on_obj.astype(int), scene_grid,scene_res,values_near_zero)
    
    crop_points_near_zero = points_near_zero[mask]
    crop_values_near_zero = values_near_zero[mask]

    obj_extract_path = "../GaussianSplattingLightning/results_here/"+this_name+"/obj_extract_pc_0.ply"
    source_pts,source_colors = load_pointcloud(obj_extract_path)

    # load obj extract gaussian point
    obj_raw_gaussian_path = "../GaussianSplattingLightning/results_here/"+this_name+"/obj_raw_gaussian_0.ply"
    obj_raw_gaussian_object = load_from_ply(path=obj_raw_gaussian_path, sh_degrees=3)

    

    #within_threshold_mask,add_surface_points_mask
    dense_gaussians = interpolate_gaussians(source_pts,source_colors,
                                            dense_points=crop_points_near_zero,
                                            dense_radius=obj_gaus_radius,
                                            dense_values = crop_values_near_zero,
                                            obj_raw_gaussian = obj_raw_gaussian_object)
    
    save_to_ply(save_path + this_name+"_results/"+"dense_obj_gaussian_0.ply",dense_gaussians)
    print("Dense Gaussian points saved to:", save_path + this_name+"_results/"+"dense_obj_gaussian_0.ply")