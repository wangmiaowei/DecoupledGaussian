
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import quaternion_to_matrix
import open3d as o3d

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def transform_shs(features, rotation_matrix):
    """
    https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
    """

    try:
        from e3nn import o3
        import einops
        from einops import einsum
    except:
        print("Please run `pip install e3nn einops` to enable SHs rotation")
        return features

    if features.shape[1] == 1:
        return features

    features = features.clone()
    shs_feat = features[:, 1:, :]

    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
    inversed_P = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=shs_feat.dtype, device=shs_feat.device)
    permuted_rotation_matrix = inversed_P @ rotation_matrix @ P
    ortho_check = torch.allclose(rotation_matrix @ rotation_matrix.transpose(-1, -2), torch.eye(3).to(rotation_matrix.device))
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

    # rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    if shs_feat.shape[1] >= 4:
        two_degree_shs = shs_feat[:, 3:8]
        two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
        two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 3:8] = two_degree_shs

        if shs_feat.shape[1] >= 9:
            three_degree_shs = shs_feat[:, 8:15]
            three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            three_degree_shs = einsum(
                D_3,
                three_degree_shs,
                "... i j, ... j -> ... i",
            )
            three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 8:15] = three_degree_shs

    return features

def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
    return torch.concatenate((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), dim=-1)

def rotate_by_wxyz_quaternions(xyz, rotations, features, quaternions: torch.tensor):
    if torch.all(quaternions == 0.) or torch.all(quaternions == torch.tensor(
            [1., 0., 0., 0.],
            dtype=quaternions.dtype,
            device=quaternions.device,
    )):
        return xyz, rotations, features

    # convert quaternions to rotation matrix
    rotation_matrix = quaternion_to_matrix(quaternions)
    # rotate xyz
    xyz = torch.matmul(xyz, rotation_matrix.T)
    # rotate gaussian quaternions
    rotations = torch.nn.functional.normalize(quat_multiply(
        rotations,
        quaternions,
    ))
    print('features.shape: ',features.shape)
    print('roatation_matrix: ',rotation_matrix.shape)
    features = transform_shs(features, rotation_matrix)
    return xyz, rotations, features

def save_ply(gaus,path):
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(this_features_dc.shape[1]*this_features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(this_features_extra.shape[1]*this_features_extra.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(this_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(this_rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    #mkdir_p(os.path.dirname(path))
    xyz = gaus[0].cuda()
    this_features_dc = gaus[1].cuda()
    this_features_extra = gaus[2].cuda()
    this_features_opacities = gaus[3].cuda()
    this_scaling =  gaus[4].cuda()
    this_rotation = gaus[5].cuda()
    print('save_path: ',path)

    xyz = xyz.detach().cpu().numpy()
    print('this_xyz.shape: ',xyz.shape)
    normals = np.zeros_like(xyz)
    f_dc = this_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = this_features_extra.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = this_features_opacities.detach().cpu().numpy()
    scale = this_scaling.detach().cpu().numpy()
    rotation = this_rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_single_ply(path):
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
    assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

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
    print('torchxyz: ',torch.tensor(xyz).shape)
    return torch.tensor(xyz,dtype=torch.float),torch.tensor(features_dc,dtype=torch.float).transpose(1, 2),torch.tensor(features_extra,dtype=torch.float).transpose(1, 2),torch.tensor(np.copy(opacities),dtype=torch.float),torch.tensor(scales,dtype=torch.float),torch.tensor(rots,dtype=torch.float)


def transform_with_vectors(gaussian_obj,r_wxyz,t_xyz=torch.tensor([0,0,0])):
    # pick properties corresponds to the specified `idx` of model
    xyz = gaussian_obj[0].cuda()
    features_dc = gaussian_obj[1].cuda()
    features_extra = gaussian_obj[2].cuda()
    features_opacities = gaussian_obj[3].cuda()
    scaling =  gaussian_obj[4].cuda()
    rotation = gaussian_obj[5].cuda()

    #print('rotation.shape: ',rotation.shape)
    #print("scaling: ",scaling.shape)
    # TODO: avoid memory copy if no rotation or scaling happened compared to previous state

    features = torch.cat((features_dc, features_extra), dim=1)  # consume a lot of memory


    # rotate
    xyz, rotation, new_features = rotate_by_wxyz_quaternions(
        xyz=xyz,
        rotations=rotation,
        features=features,
        quaternions=torch.tensor(r_wxyz).to(xyz),
    )
    #print('scaling: ',scaling)
    #print('rotations.shape: ',rotation.shape)
    """
    mask = xyz[:,2]>-3.3314
    new_features= new_features[mask]
    features_opacities = features_opacities[mask]
    scaling = scaling[mask]
    rotation =rotation[mask]
    xyz = xyz[mask]
    """
    return xyz,new_features[:, :1, :],new_features[:, 1:, :],features_opacities,scaling,rotation


if __name__ == "__main__":
    obj_name = "pillow_playroom"
    r_wxyz = torch.tensor([0.5452, -0.8383,  0.0016,  0.0000]).cuda()

    #obj_path = "/mnt/wmw/exp_res/PGSR/"+obj_name+"/point_cloud_with_obj/iteration_37000/point_cloud.ply"
    #obj_path =  "/mnt/wmw/exp_res/PGSR/" + obj_name +"/point_cloud/iteration_30000/point_cloud.ply"
    #obj_path = "/mnt/wmw/exp_res/PGSR_physgaussian/"+obj_name+"/point_cloud/iteration_30000/point_cloud.ply"
    #obj_path = "/mnt/wmw/exp_res/PGSR_gic/"+obj_name+"/dense_0.ply"
    #gaussian_obj = load_single_ply(obj_path)
    #gaus_scales = gaussian_obj[-2]
    #new_gaussian_obj = transform_with_vectors(gaussian_obj,r_wxyz,t_xyz=torch.tensor([0,0,0]))
    #save_ply(new_gaussian_obj,"/mnt/wmw/exp_res/PGSR_physgaussian/"+obj_name+"/zaxis_obj_0.ply")
    obj_path = "/mnt/wmw/exp_res/PGSR/pillow_playroom/point_cloud_with_obj_wodensepoints/iteration_37000/point_cloud.ply"
    #obj_path = "/mnt/wmw/exp_res/crop_gaussian/results_here/"+obj_name+"/obj_raw_gaussian_3.ply"
    gaussian_obj = load_single_ply(obj_path)
    gaus_scales = gaussian_obj[-2]
    new_gaussian_obj = transform_with_vectors(gaussian_obj,r_wxyz,t_xyz=torch.tensor([0,0,0]))

    save_ply(new_gaussian_obj,"/mnt/wmw/exp_res/PGSR/pillow_playroom/pillow_zaxis_wodensepoints.ply")
    #save_ply(new_gaussian_obj,"/mnt/wmw/exp_res/crop_gaussian/results_here/"+obj_name+"/zaxis_obj_3.ply")
    """
    obj_path = "/mnt/wmw/exp_res/PGSR/"+obj_name+"/point_cloud_with_obj_1/iteration_37000/point_cloud.ply"
    gaussian_obj = load_single_ply(obj_path)
    gaus_scales = gaussian_obj[-2]
    new_gaussian_obj = transform_with_vectors(gaussian_obj,r_wxyz,t_xyz=torch.tensor([0,0,0]))
    save_ply(new_gaussian_obj,"/mnt/wmw/exp_res/PGSR/"+obj_name+"/zaxis_obj_1.ply")

    obj_path = "/mnt/wmw/exp_res/PGSR/"+obj_name+"/point_cloud_with_obj_2/iteration_37000/point_cloud.ply"
    gaussian_obj = load_single_ply(obj_path)
    gaus_scales = gaussian_obj[-2]
    new_gaussian_obj = transform_with_vectors(gaussian_obj,r_wxyz,t_xyz=torch.tensor([0,0,0]))
    save_ply(new_gaussian_obj,"/mnt/wmw/exp_res/PGSR/"+obj_name+"/zaxis_obj_2.ply")


    obj_path = "/mnt/wmw/exp_res/PGSR/"+obj_name+"/point_cloud_with_obj_3/iteration_37000/point_cloud.ply"
    gaussian_obj = load_single_ply(obj_path)
    gaus_scales = gaussian_obj[-2]
    new_gaussian_obj = transform_with_vectors(gaussian_obj,r_wxyz,t_xyz=torch.tensor([0,0,0]))
    save_ply(new_gaussian_obj,"/mnt/wmw/exp_res/PGSR/"+obj_name+"/zaxis_obj_3.ply")
    """
    #scene_path = "/mnt/wmw/exp_res/PGSR/room/point_cloud_with_obj/iteration_37000/point_cloud.ply"
    #scene_path ="/mnt/wmw/exp_res/PGSR/room/zaxis_obj_0.ply"
    #scene_path =  "/mnt/wmw/exp_res/PGSR/" + obj_name +"/point_cloud/iteration_30000/point_cloud.ply"
    """"
    scene_path = "/mnt/wmw/exp_res/PGSR/"+obj_name+"/point_cloud_scene/iteration_35000/point_cloud.ply"
    scene_gaussian_scene = load_single_ply(scene_path)
    #r_wxyz = torch.tensor([ 0.4946, -0.8555, -0.1534,  0.0000]).cuda()
    new_gaussian_scene = transform_with_vectors(scene_gaussian_scene,r_wxyz,t_xyz=torch.tensor([0,0,0]))
    save_ply(new_gaussian_scene,"/mnt/wmw/exp_res/PGSR/"+obj_name+"/zaxis_obj_0.ply")
    """
    """
    # rotate_back
    scene_path = "/mnt/wmw/exp_res/PGSR_physgaussian/"+obj_name+"/PhysG_zaxis_scene_0.ply"
    scene_gaussian_scene = load_single_ply(scene_path)
    new_gaussian_scene = transform_with_vectors(scene_gaussian_scene,r_wxyz,t_xyz=torch.tensor([0,0,0]))
    save_ply(new_gaussian_scene,"/mnt/wmw/exp_res/PGSR_physgaussian/" + obj_name +"/point_cloud/iteration_33000/point_cloud.ply")
    """

#############bear###############################
#self.z_quate:  tensor([ 0.4943, -0.8554, -0.1546,  0.0000], device='cuda:0')
#bb_box 
#bounding_box:tensor([[-6.2323, -5.6572, -5.4700],
#        [ 7.4343,  7.3291,  3.0989]], device='cuda:0')
##############bonsai#############################
#self.z_quate:  tensor([ 0.4291, -0.9026, -0.0333,  0.0000]
#bouding_box: scene_bounding_box:  tensor([[-2.6067, -2.1722, -3.7594],
#[ 2.9078,  2.7349,  2.8578]], device='cuda:0')
##################bicycle Scene#########################
#self.z_quate:  tensor([ 0.7149, -0.6984,  0.0343,  0.0000]
#scene_bounding_box:  tensor([[-2.6385, -2.9545, -3.6926],
#        [ 3.0016,  2.9253,  2.6212]], device='cuda:0')
##################Kitchen scene###########################
#self.z_quate:  tensor([ 0.4703, -0.8825,  0.0106,  0.0000]
#scene_bounding_box:  tensor([[-2.9316, -3.3891, -3.9766],
#        [ 2.1855,  3.5033,  2.3477]], device='cuda:0')

##################Room################################
#self.z_quate:  tensor([ 0.8523,  0.5231, -0.0015,  0.0000]
#scene_bounding_box:  tensor([[-0.7611, -3.6143,  0.9901],
#        [ 2.5152,  0.1686,  3.6510]], device='cuda:0')

##################truck################################
#self.z_quate:  tensor([ 0.6738, -0.7382, -0.0332,  0.0000]

##################figurines############################
#self.z_quate:  tensor([ 0.6292, -0.7611, -0.1574,  0.0000], device='cuda:0')

#scene_bounding_box:  tensor([[-4.3849, -5.1414, -3.7314],
#        [ 3.1263,  2.6287,  2.9578]], device='cuda:0')


###################Playroom#########################
#self.z_quate: self.z_quate:  tensor([ 0.5452, -0.8383,  0.0016,  0.0000], device='cuda:0')
#scene_bounding_box:  tensor([[-2.5963, -0.5564, -4.1384],
#        [ 3.6470,  3.3430, -3.7425]], device='cuda:0')
######################Garden#########################
# .z_quate:  tensor([ 0.5828, -0.8126, -0.0013,  0.0000], device='cuda:0')
#scene_bounding_box:  tensor([[-2.0095, -2.0089, -2.3841],
#        [ 2.4918,  2.3780,  0.0408]], device='cuda:0')
