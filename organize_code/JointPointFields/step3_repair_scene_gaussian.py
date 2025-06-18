import pymeshlab as ml
import torch
import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import quaternion_to_matrix,matrix_to_quaternion
from pytorch3d.ops import knn_points
import argparse

# should load gaussian object
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
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
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
    

def crop_mesh_by_boundingbox(verts, faces, vertex_colors, obj_box):
    """
    Crop mesh by an axis-aligned bounding box.
    
    Args:
        verts (Tensor): (N, 3) vertex positions.
        faces (Tensor): (F, 3) triangle face indices.
        vertex_colors (Tensor): (N, 3) vertex colors.
        obj_box (Tensor): (2, 3) bounding box [min_xyz, max_xyz].

    Returns:
        filtered_verts (Tensor): vertices within bounding box.
        filtered_faces (Tensor): reindexed faces.
        filtered_vertex_colors (Tensor): colors of kept vertices.
    """
    # Create mask for vertices inside the bounding box
    mask = (verts >= obj_box[0]) & (verts <= obj_box[1])
    mask = mask.all(dim=1)

    # Filter vertices and vertex colors
    filtered_verts = verts[mask]
    filtered_vertex_colors = vertex_colors[mask]

    # Map old indices to new indices
    valid_indices = torch.nonzero(mask, as_tuple=False).squeeze()
    index_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(valid_indices)}

    # Keep only faces with all three vertices inside the box
    valid_faces_mask = mask[faces].all(dim=1)
    filtered_faces = faces[valid_faces_mask].clone()

    # Remap face indices
    for i in range(filtered_faces.shape[0]):
        for j in range(3):
            old_idx = filtered_faces[i, j].item()
            filtered_faces[i, j] = index_map[old_idx]

    return filtered_verts, filtered_faces, filtered_vertex_colors

def get_obj_bb(extract_point_cloud,test=False):
    pcd = o3d.io.read_point_cloud(extract_point_cloud)
    # Convert the point cloud points to a NumPy array
    points = np.asarray(pcd.points)

    # Compute the axis-aligned bounding box (AABB)
    aabb = pcd.get_axis_aligned_bounding_box()

    # Get the center and extend the lengths of the bounding box
    center = aabb.get_center()
    extent = aabb.get_extent()  # Lengths of the bounding box along each axis

    # Scale the bounding box by a factor of 1.2
    scale_factor = 1.2
    scaled_extent = extent * scale_factor
    # Compute the lower and upper bounds of the scaled bounding box
    obj_box_lower = center - scaled_extent / 2  # Lower bound (min corner)
    obj_box_upper = center + scaled_extent / 2  # Upper bound (max corner)

    # Combine the bounds into obj_box (as required for the mask)
    obj_box = np.array([obj_box_lower, obj_box_upper])
    return torch.tensor(obj_box)


def mesh_texture_transfer(cleaned_raw_pointcloud,dense_points,dense_faces,dense_colors,gaussian_object,
                          scale_conditioned_semantic_features):
    def prepare_scaling_rot(triangles):
        def this_dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            Project the vector v onto the subspace where vector u lies. 
            It is assumed that vector u has been normalized.
            """
            coef = this_dot(v, u)
            return coef * u
        
        """
        Approximate the covariance matrix and calculate the scaling and rotation tensors, 
        preparing the parameterized Gaussian distribution
        """
        eps = 1e-8
        v1 = triangles[:, 0].clone()
        v2 = triangles[:, 1].clone()
        v3 = triangles[:, 2].clone()
        v1 = (1/3)*(v1+v2+v3)
        _s2 = v2 - v1
        _s3 = v3 - v1
        
        # calculate the first basis vector r1
        r1 = torch.cross(_s2, _s3)
        r1 = r1 / (torch.linalg.vector_norm(r1, dim=-1, keepdim=True) + eps)
        # calculate the second basis vector r2
        s2 = torch.linalg.vector_norm(_s2, dim=-1, keepdim=True) + eps
        r2 = _s2 / s2
        # calculate the third basis vector r3
        r3 = _s3 - proj(_s3, r1) - proj(_s3, r2)
        r3 = r3 / (torch.linalg.vector_norm(r3, dim=-1, keepdim=True) + eps)

        # calculate the scaling tensor
        s3 = this_dot(_s3, r3)
        # calculate the scaling factor
        scales = torch.cat([torch.ones_like(s2) * eps,s2, s3], dim=1)
        gaussian_scaling = torch.log(scales)

        # stack the basis vectors to form the rotation matrix
        rotation = torch.stack([r1, r2, r3], dim=1)
        rotation = rotation.transpose(-2, -1)  # (B, 3, 3) shape

        # convert the rotation matrix to a quaternion
        gaussian_rotation = matrix_to_quaternion(rotation)
        return v1,r1,gaussian_scaling,gaussian_rotation
    
    triangles = dense_points[dense_faces]
    gaus_center,gaus_normals,gaus_scales,gaus_rots = prepare_scaling_rot(triangles)

    #########################  Scene Raw Gaussian #############

    points,features_dc,features_rest,opacities,scales,rots = gaussian_object

    points = torch.tensor(points,dtype=torch.float).cuda()
    scales = torch.tensor(scales,dtype=torch.float).cuda()
    rots = torch.tensor(rots,dtype=torch.float).cuda()

    opacities = torch.tensor(opacities.copy(),dtype=torch.float).cuda()
    features_dc = torch.tensor(features_dc,dtype=torch.float).reshape(len(points),-1).cuda()
    features_rest = torch.tensor(features_rest,dtype=torch.float).cuda()

    raw_knn_result = knn_points(torch.tensor(cleaned_raw_pointcloud).unsqueeze(0).to(torch.float32).cuda(), points.unsqueeze(0).cuda(), K=1)
    raw_idx_nearest = (raw_knn_result.idx.squeeze(0))[:,0]
    raw_points = cleaned_raw_pointcloud
    raw_scales = scales[raw_idx_nearest]
    raw_rots = rots[raw_idx_nearest]
    raw_opacities = opacities[raw_idx_nearest]
    raw_feature_dc = features_dc[raw_idx_nearest][:,None,:]
    raw_features_rest = (features_rest[raw_idx_nearest]).transpose(2,1)


    K = 100  # number of nearest neighbors
    knn_result = knn_points(torch.tensor(gaus_center).unsqueeze(0).to(torch.float32), points.unsqueeze(0).cuda(), K=K)
    # Extract the index and distance of the nearest neighbor
    idx_nearest = knn_result.idx.squeeze(0)  # Nearest Neighbor Index, shape = (M, K)
    dists_nearest = knn_result.dists.squeeze(0)  # Nearest Neighbor Distance, shape = (M, K)

    mean_pnts = knn_points(torch.tensor(points).unsqueeze(0).to(torch.float32).cuda(), points.unsqueeze(0).cuda(), K=2)
    pnts_dists_nearest = (mean_pnts.dists.squeeze(0))[:,1].mean()
    mask = dists_nearest[:,0] >0# 0.1*pnts_dists_nearest

    
    # Step 2: Calculate the weights using inverse distance weighted interpolation (IDW)
    sigma = 10.0
    weights = torch.exp(-dists_nearest / (2 * sigma ** 2)) 
    weights = weights / weights.sum(dim=1, keepdim=True)  # Normalized weights, shape = (M, K)
    # Define an interpolation function to perform weighted averaging for any feature.
    def interpolate_features(features, idx_nearest, weights):
        nearest_features = features[idx_nearest]  # Obtain the features of the nearest neighbors, with shape = (M, K, F)
        interpolated_features = (nearest_features * weights.unsqueeze(-1)).sum(dim=1)  # shape = (M, F)
        return interpolated_features
    new_scale_semantic = interpolate_features(scale_conditioned_semantic_features.cuda(), idx_nearest, weights)

    def inverse_sigmoid(x):
        return torch.log(x/(1-x))
    gaus_features_dc = interpolate_features(features_dc.cuda(), idx_nearest, weights)[:,None,:] #.view(len(dense_points),3,1)  # (M, F)
    gaussian_opacities = inverse_sigmoid(0.9*torch.ones(len(gaus_features_dc),1).cuda())
    gaus_features_rest = interpolate_features(features_rest.cuda().reshape(len(features_rest),-1), idx_nearest, weights)
    gaus_features_rest = gaus_features_rest.reshape(len(gaus_features_rest),-1,3)
    gaus_center = gaus_center[mask].cpu()#torch.cat([raw_points,gaus_center[mask]],dim=0).to(torch.float32).cpu()
    gaus_normals = gaus_normals[mask].cpu()#torch.cat([raw_normals,gaus_normals[mask]],dim=0).to(torch.float32).cpu()
    gaussian_opacities = gaussian_opacities[mask].cpu()#torch.cat([raw_opacities,gaussian_opacities[mask]],dim=0).to(torch.float32).cpu()
    gaus_features_dc = gaus_features_dc[mask].cpu()
    gaus_features_rest = torch.zeros(gaus_features_rest[mask].cpu().shape)#torch.cat([raw_features_rest,gaus_features_rest[mask]],dim=0).to(torch.float32).cpu()
    gaus_scales = gaus_scales[mask].cpu()#torch.cat([raw_scales,gaus_scales[mask]],dim=0).to(torch.float32).cpu()
    gaus_rots = gaus_rots[mask].cpu()#torch.cat([raw_rots,gaus_rots[mask]],dim=0).to(torch.float32).cpu()
    real_scene_gaussian = (gaus_center,gaus_normals,
            gaussian_opacities,gaus_features_dc,
            gaus_features_rest,gaus_scales,
            gaus_rots)

    gaus_center1 = raw_points#.cpu()#torch.cat([raw_points,gaus_center[mask]],dim=0).to(torch.float32).cpu()
    gaus_normals1 = torch.zeros(gaus_center1.shape)#torch.cat([raw_normals,gaus_normals[mask]],dim=0).to(torch.float32).cpu()
    gaussian_opacities1 = raw_opacities.cpu()#torch.cat([raw_opacities,gaussian_opacities[mask]],dim=0).to(torch.float32).cpu()
    gaus_rots1 = raw_rots.cpu()#torch.cat([raw_rots,gaus_rots[mask]],dim=0).to(torch.float32).cpu()
    gaus_scales1 = raw_scales.cpu()#torch.cat([raw_scales,gaus_scales[mask]],dim=0).to(torch.float32).cpu()

    gaus_features_dc1 = raw_feature_dc.cpu()#torch.cat([raw_feature_dc,gaus_features_dc[mask]],dim=0).to(torch.float32).cpu()
    gaus_features_rest1 = raw_features_rest.cpu()#torch.cat([raw_features_rest,gaus_features_rest[mask]],dim=0).to(torch.float32).cpu()
    
    scene_gaussian= (gaus_center1,gaus_normals1,
            gaussian_opacities1,gaus_features_dc1,
            gaus_features_rest1,gaus_scales1,
            gaus_rots1)

    return scene_gaussian,real_scene_gaussian, new_scale_semantic[mask]


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

    xyz = gaussians[0]#.numpy()
    normals = gaussians[1]#.numpy()
    opacities = gaussians[2]
    f_dc = torch.tensor(gaussians[3]).flatten(start_dim=1).contiguous().cpu().numpy()#.numpy() #N*3*1
    f_rest = torch.tensor(gaussians[4]).flatten(start_dim=1).contiguous().cpu().numpy() # N*3*15
    scale = gaussians[5]#.numpy()
    rotation = gaussians[6]#.numpy()


    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("object_name", type=str, help="object name, e.g., bear")
    args = parser.parse_args()
    obj_name = args.object_name
    ms = ml.MeshSet()

    obj_name = "bear"
    folder_path ="../exp_res/" + obj_name + "/dense_poisson_scripts/"
    scene_recons_path =folder_path +  obj_name+"_results"+"/scene_mesh.ply"


    scene_raw_pointcloud = "../GaussianSplattingLightning/results_here/" +obj_name + "/scene_point_cloud.ply"
    scene_raw_gaussian = "../GaussianSplattingLightning/results_here/" + obj_name + "/scene_raw_gaussian.ply"
    scene_raw_gaussian_segment = "../GaussianSplattingLightning/results_here/" + obj_name + "/scene_segment_features_gaussian.pt"

    save_patch_path = folder_path + obj_name+"_results/"+"patch_segment_real.ply"
    save_raw_patch_path = folder_path + obj_name+"_results/"+"patch_segment.ply"
    cleaned_raw_pointcloud2 = o3d.io.read_point_cloud(scene_raw_pointcloud)
    gaussian_object = load_from_ply(path=scene_raw_gaussian, sh_degrees=3)
    mesh = o3d.io.read_triangle_mesh(scene_recons_path)

    verts2 = torch.tensor(np.asarray(mesh.vertices))
    faces2 = torch.tensor(np.asarray(mesh.triangles))
    vertex_colors2 = torch.tensor(np.asarray(mesh.vertex_colors))


    extract_obj_point_cloud_path = "../GaussianSplattingLightning/results_here/" + obj_name+"/obj_extract_pc_0"+".ply"
    obj_box = get_obj_bb(extract_obj_point_cloud_path)

    verts,faces,vertex_colors = crop_mesh_by_boundingbox(verts2,faces2,vertex_colors2,obj_box)
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=verts.numpy().astype(np.float64),
                    face_matrix=faces.numpy().astype(np.int32)))
    ms.meshing_isotropic_explicit_remeshing(targetlen=ml.PercentageValue(0.3))
    m = ms.current_mesh()

    verts = torch.tensor(m.vertex_matrix()).to(torch.float32)
    faces = torch.tensor(m.face_matrix())

    scale_conditioned_semantic_features = torch.load(scene_raw_gaussian_segment)

    cleaned_raw_pointcloud = np.asarray(cleaned_raw_pointcloud2.points)

    cropped_scene_raw_gaussian,real_fix_patch_gaussian,new_scale_semantic = mesh_texture_transfer(cleaned_raw_pointcloud,verts.cuda(),faces.cuda(),vertex_colors.cuda(),gaussian_object,scale_conditioned_semantic_features)

    save_to_ply(save_raw_patch_path,cropped_scene_raw_gaussian)
    save_to_ply(save_patch_path,real_fix_patch_gaussian)
    print(f"Processing complete. Output saved at: {save_patch_path}")


    # This part is commented out because it is to be for extracting multiple patches from the object.
    # Uncomment if you want to extract multiple patches
    # real_fix_patch_gaussians = []
    # for i in range(4):
    #     extract_obj_point_cloud_path = "/mnt/wmw/exp_res/crop_gaussian/results_here/"+obj_name+"/obj_extract_pc_"+str(i)+".ply"
    #     if i==0:
    #         obj_box = get_obj_bb(extract_obj_point_cloud_path,test=True)
    #     else:
    #         obj_box = get_obj_bb(extract_obj_point_cloud_path)
    #     verts,faces,vertex_colors = crop_mesh_by_boundingbox(verts2,faces2,vertex_colors2,obj_box)
    #     ms = ml.MeshSet()
    #     ms.add_mesh(ml.Mesh(vertex_matrix=verts.numpy().astype(np.float64),
    #                     face_matrix=faces.numpy().astype(np.int32)))
    #     #ms.save_current_mesh("mesh0.ply")
    #     ms.meshing_isotropic_explicit_remeshing(targetlen=ml.PercentageValue(0.3))
    #     m = ms.current_mesh()

    #     verts = torch.tensor(m.vertex_matrix()).to(torch.float32)
    #     faces = torch.tensor(m.face_matrix())

    #     scale_conditioned_semantic_features = torch.load(scene_raw_gaussian_segment)
        
    #     cleaned_raw_pointcloud = np.asarray(cleaned_raw_pointcloud2.points)
        
    #     cropped_scene_raw_gaussian,real_fix_patch_gaussian,new_scale_semantic = mesh_texture_transfer(cleaned_raw_pointcloud,verts.cuda(),faces.cuda(),vertex_colors.cuda(),gaussian_object,scale_conditioned_semantic_features)

    #     real_fix_patch_gaussians.append(real_fix_patch_gaussian)

    # save_to_ply(save_raw_patch_path,cropped_scene_raw_gaussian)

    # real_fix_patch_gaussian = (torch.cat([real_fix_patch_gaussians[0][0],real_fix_patch_gaussians[1][0],
    #                                     real_fix_patch_gaussians[2][0],real_fix_patch_gaussians[3][0]]),
    #                             torch.cat([real_fix_patch_gaussians[0][1],real_fix_patch_gaussians[1][1],
    #                                     real_fix_patch_gaussians[2][1],real_fix_patch_gaussians[3][1]]),
    #                             torch.cat([real_fix_patch_gaussians[0][2],real_fix_patch_gaussians[1][2],
    #                                     real_fix_patch_gaussians[2][2],real_fix_patch_gaussians[3][2]]),
    #                             torch.cat([real_fix_patch_gaussians[0][3],real_fix_patch_gaussians[1][3],
    #                                     real_fix_patch_gaussians[2][3],real_fix_patch_gaussians[3][3]]),
    #                             torch.cat([real_fix_patch_gaussians[0][4],real_fix_patch_gaussians[1][4],
    #                                     real_fix_patch_gaussians[2][4],real_fix_patch_gaussians[3][4]]),
    #                             torch.cat([real_fix_patch_gaussians[0][5],real_fix_patch_gaussians[1][5],
    #                                     real_fix_patch_gaussians[2][5],real_fix_patch_gaussians[3][5]]),
    #                             torch.cat([real_fix_patch_gaussians[0][6],real_fix_patch_gaussians[1][6],
    #                                     real_fix_patch_gaussians[2][6],real_fix_patch_gaussians[3][6]]))
    # save_to_ply(save_patch_path,real_fix_patch_gaussian)

