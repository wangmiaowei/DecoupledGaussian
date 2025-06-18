import torch
import torch.nn.functional as F
import pytorch3d
from pytorch3d.io import IO
import point_cloud_utils as pcu
from simple_knn._C import distCUDA2
from pytorch3d.ops import knn_points
import open3d as o3d
import numpy as np
from PIL import Image
import torchvision
import os
import copy
import matplotlib.pyplot as plt
from matplotlib import cm

import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer

from pytorch3d.transforms import quaternion_to_matrix,matrix_to_quaternion
#import nksr
#obj_rec = nksr.Reconstructor(torch.device("cuda:0"))
import torch

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
def crop_mesh_by_mask(xyz,xyz_colors,faces,obj_mask):
    xyz_cropped = xyz[obj_mask]
    xyz_colors_cropped = xyz_colors[obj_mask]
    # 创建原始顶点索引到新顶点索引的映射
    # 首先需要找出被保留的顶点的原始索引
    original_indices = torch.nonzero(obj_mask).squeeze()
    # 创建从旧顶点索引到新顶点索引的映射表
    index_mapping = -torch.ones(xyz.shape[0], dtype=torch.long, device=xyz.device)
    index_mapping[original_indices] = torch.arange(xyz_cropped.shape[0], device=xyz.device)
    valid_faces_mask = (obj_mask[faces[:, 0]] & obj_mask[faces[:, 1]] & obj_mask[faces[:, 2]])
    faces_masked = faces[valid_faces_mask]
    # 更新面片，只保留所有顶点都在 mask 中的面片
    # 更新面片索引，使用新顶点索引替换旧顶点索引
    faces_masked = index_mapping[faces_masked]

    cropped_mesh = o3d.geometry.TriangleMesh()
    cropped_mesh.vertices = o3d.utility.Vector3dVector(xyz_cropped.cpu().numpy())
    cropped_mesh.triangles = o3d.utility.Vector3iVector(faces_masked.cpu().numpy())
    cropped_mesh.vertex_colors = o3d.utility.Vector3dVector(xyz_colors_cropped.cpu().numpy())

    return cropped_mesh

def save_crop_object(mesh_pts,obj_mask,mesh_features,output_path,extract_mesh_nums):

    xyz = torch.tensor(np.asarray(mesh_pts.points), dtype=torch.float)[obj_mask]
    normals  = torch.tensor(np.asarray(mesh_pts.normals), dtype=torch.float)[obj_mask]
    colors  = torch.tensor(np.asarray(mesh_pts.colors), dtype=torch.float)[obj_mask]
    seg_features = mesh_features[obj_mask]

    # Create Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    # Set points, normals, and colors
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)


    obj_filename = f"obj_extract_pc_{extract_mesh_nums}.ply"
    obj_path = os.path.join(output_path, obj_filename)
    o3d.io.write_point_cloud(obj_path, point_cloud)
    
    points = torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32).unsqueeze(0).cuda()
    # Step 2: 使用 PyTorch3D 的 farthest point sampling 采样一万个点
    # pytorch3d.ops.farthest_point_sample expects inputs of shape [B, N, 3]
    #num_samples = 10000  # 你需要采样的点数

    # farthest_point_sample 返回选中点的索引，形状为 [B, num_samples]

    #farthest_point_idx, _ = pytorch3d.ops.sample_farthest_points(points, K=num_samples)
    # 将 farthest_point_idx 转换为 long 类型
    #farthest_point_idx = farthest_point_idx.long()

    # Step 3: 使用索引从原始点云中提取采样点
    # farthest_point_idx 的形状是 [B, num_samples]，所以我们 squeeze(0) 去掉批次维度
    #sampled_points = points[0, farthest_point_idx.squeeze(0)]  # 提取采样的点，形状为 [num_samples, 3]
    # 将张量组合到一个字典中
    data_dict = {
        "xyz": xyz,
        "normals": normals,
        "colors": colors,
        "seg_features": seg_features
    }
    # 保存为一个 .pt 文件
    obj_filename2 = f"obj_extract_pc_w_segfeatures_{extract_mesh_nums}.pt"
    obj_path = os.path.join(output_path, obj_filename2)
    torch.save(data_dict, obj_path)
    print("finised the save of single object")
    #return sampled_points
def save_crop_meshV2(obj_mesh,obj_mask):
    xyz = torch.tensor(np.asarray(obj_mesh.vertices), dtype=torch.float)
    xyz_colors = torch.tensor(np.asarray(obj_mesh.vertex_colors), dtype=torch.float)
    faces = torch.tensor(np.asarray(obj_mesh.triangles), dtype=torch.long)

    cropped_mesh = crop_mesh_by_mask(xyz,xyz_colors,faces,obj_mask)
    left_cropped_mesh = crop_mesh_by_mask(xyz,xyz_colors,faces,~obj_mask)
    output_path = "./GaussianSplattingLightning/results_here"
    o3d.io.write_triangle_mesh(os.path.join(output_path, "tsdf_fusion_post_cropped.ply"),cropped_mesh)
    o3d.io.write_triangle_mesh(os.path.join(output_path, "tsdf_fusion_post_left_cropped.ply"),left_cropped_mesh)
    print('crop_mesh_finished!')


def transport_property_from_gaussian2objpc(obj_pc,pc,scale_conditioned_semantic_features):
    #print('pc.shape: ',pc.shape) #N*3
    #print('scale_conditioned_semantic_features: ',scale_conditioned_semantic_features.shape)#N*3
    xyz = torch.tensor(np.asarray(obj_pc.points), dtype=torch.float).cuda()
    knn_result = knn_points(xyz.unsqueeze(0), pc.unsqueeze(0), K=1)  # K=1 for the nearest neighbor
    nearest_indices = knn_result.idx.squeeze(0)
    nearest_features = scale_conditioned_semantic_features[nearest_indices.cpu().squeeze(1)]  # Shape: (V, 3)
    return nearest_features

def transport_property_from_gaussian2mesh(pc,scale_conditioned_semantic_features):
    #print('pc.shape: ',pc.shape) #N*3
    #print('scale_conditioned_semantic_features: ',scale_conditioned_semantic_features.shape)#N*3
    output_path = "./GaussianSplattingLightning/results_here"
    mesh = o3d.io.read_triangle_mesh(os.path.join(output_path, "tsdf_fusion_post.ply"))
    xyz = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float).cuda()
    knn_result = knn_points(xyz.unsqueeze(0), pc.unsqueeze(0), K=1)  # K=1 for the nearest neighbor
    nearest_indices = knn_result.idx.squeeze(0)
    nearest_features = scale_conditioned_semantic_features[nearest_indices.cpu().squeeze(1)]  # Shape: (V, 3)
    print('nearst_features: ',nearest_features.shape)
    return mesh,nearest_features
def crop_mesh(obj_bounding_box,z_quate):
    # convert quaternions to rotation matrix
    output_path = "./GaussianSplattingLightning/results_here"
    mesh = o3d.io.read_triangle_mesh(os.path.join(output_path, "tsdf_fusion_post.ply"))
    # 获取顶点坐标
    xyz = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float)

    xyz_colors = torch.tensor(np.asarray(mesh.vertex_colors), dtype=torch.float)

    rotation_matrix = torch.tensor(qvec2rotmat(z_quate.cpu().numpy()), dtype=torch.float, device=xyz.device)
    # rotate xyz
    xyz = torch.matmul(xyz, rotation_matrix.T)

    # 根据 bounding box 截取 xyz_rotated 中的顶点
    obj_bounding_box = obj_bounding_box.to(xyz.device)
    bounding_box_mask = ((xyz>=obj_bounding_box[0])&(xyz<=obj_bounding_box[1])).all(dim=1)
    xyz_cropped = xyz[bounding_box_mask]
    xyz_colors_cropped = xyz_colors[bounding_box_mask]

    faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.long)
    
    # 创建原始顶点索引到新顶点索引的映射
    # 首先需要找出被保留的顶点的原始索引
    original_indices = torch.nonzero(bounding_box_mask).squeeze()
    # 创建从旧顶点索引到新顶点索引的映射表
    index_mapping = -torch.ones(xyz.shape[0], dtype=torch.long, device=xyz.device)
    index_mapping[original_indices] = torch.arange(xyz_cropped.shape[0], device=xyz.device)
    
    # 更新面片，只保留所有顶点都在 mask 中的面片
    valid_faces_mask = (bounding_box_mask[faces[:, 0]] & bounding_box_mask[faces[:, 1]] & bounding_box_mask[faces[:, 2]])
    
    faces_masked = faces[valid_faces_mask]

    # 更新面片索引，使用新顶点索引替换旧顶点索引
    faces_masked = index_mapping[faces_masked]

    cropped_mesh = o3d.geometry.TriangleMesh()
    cropped_mesh.vertices = o3d.utility.Vector3dVector(xyz_cropped.cpu().numpy())
    cropped_mesh.triangles = o3d.utility.Vector3iVector(faces_masked.cpu().numpy())
    cropped_mesh.vertex_colors = o3d.utility.Vector3dVector(xyz_colors_cropped.cpu().numpy())
    

    o3d.io.write_triangle_mesh(os.path.join(output_path, "tsdf_fusion_post_cropped.ply"),cropped_mesh)
    print('crop_mesh_finished!')

def rasterize_img_depth_nms(segment_mask, pc,viewpoint_cameras,extract_nums):
    voxel_size = 0.002
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size,
    sdf_trunc=4.0*voxel_size,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    #opacities = opacities * self.segment_mask.unsqueeze(-1)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    bg_color = [0,0,0] 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    idxes = 0
    for viewpoint_camera in viewpoint_cameras:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        print('start_to_process_view num: ', idxes)
        raster_settings = PlaneGaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=1.0,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=1,#pc.active_sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                render_geo=True,
                debug=False)
        rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)
        global_normal = pc.get_normal(viewpoint_camera.camera_center)
        local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
        pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] +  viewpoint_camera.world_view_transform[3,:3]
        depth_z = pts_in_cam[:, 2]
        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance
        colors_precomp =  None
        cov3D_precomp = None

        shs = pc.get_features
        rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,# * segment_mask,
            scales = scales,
            rotations = rotations,
            all_map = input_all_map,
            cov3D_precomp = cov3D_precomp)
        if plane_depth.max()==0:
            continue
        rendered_imageV2, _, _, _, plane_depthV2 = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity* segment_mask,
            scales = scales,
            rotations = rotations,
            all_map = input_all_map,
            cov3D_precomp = cov3D_precomp)

        
        rendered_normal = out_all_map[0:3]
        rendered_alpha = out_all_map[3:4, ]
        rendered_distance = out_all_map[4:5, ]
        rendering = rendered_image.clamp(0.0, 1.0)
        #torchvision.utils.save_image(rendering, os.path.join("/home/wmw/project/codespace/gaussian-splatting-lightning/results_here/render_path", str(idxes) + "_full.png"))
        rendered_imageV2 = rendered_imageV2.clamp(0.0,1.0).permute(1, 2, 0).detach().cpu().numpy()
        #torchvision.utils.save_image(rendered_imageV2, os.path.join("/home/wmw/project/codespace/gaussian-splatting-lightning/results_here/render_path", str(idxes) + ".png"))
        #torchvision.utils.save_image(rendering - rendered_imageV2, os.path.join("/home/wmw/project/codespace/gaussian-splatting-lightning/results_here/render_path", str(idxes) + "_residual.png"))
        gray_image = np.dot(rendered_imageV2[..., :3], [0.2989, 0.5870, 0.1140])
        mask = (gray_image > 0)
        #mask_uint8 = (mask > 0).astype(np.uint8) * 255

        # Create an image from the mask
        #mask_image = Image.fromarray(mask_uint8, mode='L')

        # Save the image
        #mask_image.save(os.path.join("/home/wmw/project/codespace/gaussian-splatting-lightning/results_here/render_path", str(idxes) + "_mask.png"))


        _, H, W = rendering.shape
        depth = plane_depth.squeeze()

        depth_tsdf = depth.clone()
        depth_tsdf[depth_tsdf > 5] = 0
        #depth_tsdf[(plane_depthV2.squeeze()==0)] = 0
                
        depth_tsdf = depth_tsdf.detach().cpu().numpy()
        mask = mask & (plane_depthV2.squeeze().detach().cpu().numpy()>0)
        """
        # 第一步：找到所有非零点的坐标
        non_zero_coords = np.argwhere(mask > 0)
        # 第二步：找到最小和最大的边界值
        min_y, min_x = np.min(non_zero_coords, axis=0)
        max_y, max_x = np.max(non_zero_coords, axis=0)

        # 计算矩形中心点
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2

        # 计算原始矩形的宽度和高度
        height = max_y - min_y
        width = max_x - min_x

        # 扩展 1.5 倍后的宽度和高度
        new_height = int(height * 1.5)
        new_width = int(width * 1.5)

        # 计算扩展后的新边界，确保不超出图像边界
        mask_height, mask_width = mask.shape
        new_min_y = max(0, center_y - new_height // 2)
        new_max_y = min(mask_height - 1, center_y + new_height // 2)
        new_min_x = max(0, center_x - new_width // 2)
        new_max_x = min(mask_width - 1, center_x + new_width // 2)

        # 创建一个新的全零 mask，然后设置扩展的矩形区域为 1
        rectangle_mask = np.zeros_like(mask)
        rectangle_mask[new_min_y:new_max_y+1, new_min_x:new_max_x+1] = 1
        """

        depth_tsdf[mask==0] = 0
        
        pose = np.identity(4)

        pose[:3,:3] = viewpoint_camera.R.transpose(-1,-2)
        pose[:3, 3] = viewpoint_camera.T

        color = (rendering.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        color = o3d.geometry.Image(np.ascontiguousarray(color))
        depth = o3d.geometry.Image((depth_tsdf*1000).astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)


        # 保存彩色图像
        """
        # 假设 `rgbd` 是您的 RGBDImage
        color_image = rgbd.color
        depth_image = rgbd.depth
        path = "/home/wmw/project/codespace/gaussian-splatting-lightning/results_here/RGBD_kitchen/"
        o3d.io.write_image(path+str(idxes)+"_RGB.png", color_image)

        # 将 depth_image 转换为 NumPy 数组
        depth_image_np = np.asarray(depth_image)

        # 归一化深度图像以适应可视化
        depth_min, depth_max = depth_image_np.min(), depth_image_np.max()
        depth_normalized = (depth_image_np - depth_min) / (depth_max - depth_min)  # 归一化到0-1

        # 使用伪彩色（colormap）显示深度
        colormap = cm.turbo  # 可以选择其他 colormap，比如 'plasma', 'inferno'
        depth_colored = colormap(depth_normalized)  # 应用 colormap
        depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)  # 转换为RGB格式

        # 保存伪彩色深度图像
        plt.imsave(path + str(idxes) + "_depth_colored.png", depth_colored)
        """

        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(W, H,viewpoint_camera.Fx, 
                                            viewpoint_camera.Fy, 
                                            viewpoint_camera.Cx,
                                            viewpoint_camera.Cy),pose)
        idxes = idxes +1
    print('extract mesh from volume')
    mesh = volume.extract_triangle_mesh()
    pointcloud  = volume.extract_point_cloud()
    output_file = "extracted_pc.ply"
    #wait for me!!!!!
    # Save the point cloud to the specified file
    o3d.io.write_point_cloud(output_file, pointcloud)
    #path = os.path.join(dataset.model_path, "mesh")
    #os.makedirs(path, exist_ok=True)
    #output_path = "/home/wmw/project/codespace/gaussian-splatting-lightning/results_here"
    #o3d.io.write_triangle_mesh(os.path.join(output_path, "tsdf_fusion.ply"), mesh, 
    #                            write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
    mesh = clean_mesh(mesh)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    print('clean mesh solved and estimate the normals')
    print('How many dense points prepared to extract: ',extract_nums)
    mesh_pts = mesh.sample_points_uniformly(number_of_points=extract_nums)
    #mesh_pts = o3d.geometry.TriangleMesh.sample_points_poisson_disk(mesh, number_of_points=extract_nums)#20000
    #o3d.io.write_triangle_mesh(os.path.join(output_path, "tsdf_fusion_post.ply"), mesh, 
    #                            write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
    print('extract_coarse_point_cloud successful')
    return mesh_pts


def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0
def quaternion_rotate_vector(q, v):
    # q 是一个四元数 [w, x, y, z]
    # v 是要旋转的向量 [v_x, v_y, v_z]
    
    # 提取四元数的标量和向量部分
    w, x, y, z = q
    q_v = torch.tensor([x, y, z])

    # 计算四元数的共轭
    q_conjugate = torch.tensor([w, -x, -y, -z])

    # 将向量 v 转换为纯四元数 (0, v_x, v_y, v_z)
    v_quaternion = torch.cat((torch.tensor([0.0]), v))

    # 使用四元数旋转公式: q * v_quaternion * q^-1
    # 首先计算 q * v_quaternion
    temp = quat_multiply(q, v_quaternion)
    # 然后计算结果乘以 q^-1
    v_rotated_quaternion = quat_multiply(temp, q_conjugate)

    # 提取旋转后的向量部分
    v_rotated = v_rotated_quaternion[1:]  # v_rotated_quaternion 的前一部分是标量部分，后面是向量部分

    return v_rotated

def quat_multiply(q1, q2):
    # q1 和 q2 都是四元数 [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.tensor([w, x, y, z])
"""
# 示例四元数和向量
q = torch.tensor([0.7071, 0.7071, 0.0, 0.0])  # 90度绕x轴的旋转
v = torch.tensor([0.0, 1.0, 0.0])            # 沿y轴的向量

# 旋转向量
v_rotated = quaternion_rotate_vector(q, v)
print("Rotated vector:", v_rotated)
"""


def remove_outliers_scene_p2g(means,segment_mask,grid_num=20):
    # 提取 segment_mask 区域的点坐标
    new_mask = torch.zeros_like(segment_mask)
    cropped_points = means[segment_mask]

    min_bound = cropped_points.min(dim=0)[0]
    max_bound = cropped_points.max(dim=0)[0]
    grid_size = (max_bound - min_bound)/grid_num

    center_bound = 0.5*(min_bound+max_bound)
    dist_small = 0.1*(center_bound-min_bound)
    
    min_bound = min_bound - dist_small
    max_bound = max_bound + dist_small
    
    in_box_mask = ((means>=min_bound)&(means<=max_bound)).all(dim=1)
    means_in_box = means[in_box_mask&(~segment_mask)]
    # print grid coods:
    grid_coords = ((means_in_box-min_bound)/grid_size).floor().long()
    # 将 segment_mask 的点映射到网格中的位置
    cropped_grid_coords = ((cropped_points - min_bound) / grid_size).floor().long()
    def to_linear_index(coords, grid_num):
        """将网格坐标转换为线性索引"""
        return coords[:, 0] * grid_num * grid_num + coords[:, 1] * grid_num + coords[:, 2]#
    linear_cropped_indices = to_linear_index(cropped_grid_coords, grid_num)
    linear_means_indices = to_linear_index(grid_coords, grid_num)
    # 判断哪些网格包含了 cropped_points
    is_in_cropped_grid = torch.isin(linear_means_indices, linear_cropped_indices)#true false
    new_mask[in_box_mask&(~segment_mask)] = is_in_cropped_grid
    return new_mask

def quaternion_zaxis_estimation(pts,nms):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy())
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    plane_nms=torch.tensor(plane_model[:3]).cuda().float()
    plane_raw_nms = nms[inliers].mean(dim=0).float()
    if plane_nms@plane_raw_nms <0:
        plane_nms = -1*plane_nms
    
    k = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda()# z 轴单位向量
    v = torch.cross(plane_nms, k)
    v_norm = torch.norm(v)
    # 如果法向量已经与 z 轴平行，直接返回
    if v_norm.item() == 0:
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).cuda()
    else:
        # 计算旋转角度 (theta)
        theta = torch.acos(torch.dot(plane_nms, k) / torch.norm(plane_nms))
        # 归一化旋转轴
        v = v / v_norm
        # 计算四元数
        w = torch.cos(theta / 2.0)
        sin_theta_over_2 = torch.sin(theta / 2.0)
        x = v[0] * sin_theta_over_2
        y = v[1] * sin_theta_over_2
        z = v[2] * sin_theta_over_2
        quaternion = torch.tensor([w, x, y, z]).cuda()
    return quaternion
def incremental_knn(masked_points, k=5, chunk_size=512):
    """
    Compute mean distances of k-nearest neighbors using PyTorch3D's knn_points.
    
    Args:
        masked_points (torch.Tensor): Input point cloud of shape (N, 3).
        k (int): Number of nearest neighbors to find.
        chunk_size (int): Chunk size to process points incrementally to avoid memory issues.
        
    Returns:
        mean_distances (torch.Tensor): Mean distances of the k-nearest neighbors for each point.
    """
    # Ensure input is in the correct shape
    if masked_points.dim() == 2:
        masked_points = masked_points.unsqueeze(0)  # Add batch dimension if needed

    # Split masked points into chunks to handle large datasets
    chunks = torch.split(masked_points, chunk_size, dim=1)
    all_mean_distances = []

    for chunk in chunks:
        # Compute k-NN
        knn_result = knn_points(chunk, chunk, K=k, return_nn=True)
        dists = knn_result.dists.squeeze(0)  # dists shape: (chunk_size, k)
        
        # Compute mean distance for each point
        mean_dists = dists.mean(dim=-1)  # Mean of distances along the k dimension
        all_mean_distances.append(mean_dists)

    # Concatenate mean distances from all chunks
    mean_distances = torch.cat(all_mean_distances, dim=0)
    return mean_distances
def remove_outliers_object(point_cloud, mask,ratios, k=10):
    """
    从点云中根据掩码选取点，使用k近邻距离移除离群点，并返回经过移除后的点云在原始点云中的掩码和索引。

    参数:
    - point_cloud: (N, 3) 的PyTorch张量，表示原始点云。需在GPU上。
    - mask: (N,) 的布尔类型PyTorch张量，表示选取点的掩码。需在GPU上。
    - k: k近邻的数量，用于计算每个点的局部密度。
    - threshold: 离群点移除的距离阈值。

    返回:
    - new_mask: 原始点云中经过离群点移除后的掩码。
    - indices: 原始点云中经过离群点移除后的点的索引。
    """
    
    # 确保输入张量在GPU上
    assert point_cloud.is_cuda and mask.is_cuda, "point_cloud and mask must be on GPU"

    # 从掩码中提取有效点
    masked_points = point_cloud[mask]  # (M, 3)
    ratios_points = ratios[mask]

    mean_distances = incremental_knn(masked_points, k=k, chunk_size=512)
    
    """
    dist_matrix = torch.cdist(masked_points, masked_points)  # (M, M)
    
    # 找到每个点的k个最近邻的距离
    knn_distances, _ = torch.topk(dist_matrix, k=k, largest=False)  # (M, k)

    # 计算平均距离
    mean_distances = knn_distances.mean(dim=1)  # (M,)
    """
    #mean_distances = distCUDA2(masked_points).float().cuda()
    
    mean = mean_distances.mean()
    std = mean_distances.std()
    ratio_mean = ratios_points.mean()
    ratio_std = ratios_points.std()
    # 标记哪些点是离群点（距离大于阈值）
    inlier_mask = (mean_distances-mean) > 3*std  # (M,)
    # ratio is larger:
    inlier_mask_ratio = (ratios_points-ratio_mean) > 3*ratio_std
    # 计算经过移除后的新掩码
    new_mask = torch.zeros_like(mask)
    #print('inlier_mask: ',inlier_mask.shape)
    #print('inlier_mask_ratio: ',inlier_mask_ratio.shape)
    # 计算在mask的掩码
    raw_mask = torch.logical_or(inlier_mask,inlier_mask_ratio)
    inlier_indices = mask.nonzero(as_tuple=True)[0][raw_mask]
    #inlier_indices = mask.nonzero(as_tuple=True)[0][inlier_mask]
    new_mask[inlier_indices] = True
    return new_mask
def estimate_object_shape(means_obj,normals_obj):
    print('store_single_object')
    pcu.save_mesh_vn("./GaussianSplattingLightning/results_here/this.ply", 
                     means_obj.detach().cpu().numpy(), normals_obj.detach().cpu().numpy())
    #field = obj_rec.reconstruct(means_obj, normals_obj,detail_level=1.0)
    #mesh = field.extract_dual_mesh(mise_iter=1)
    #pytorch3d.io.save_ply("/home/wmw/project/codespace/gaussian-splatting-lightning/results_here/donuts.ply" ,mesh.v.cpu(),mesh.f.cpu())
def crop_mesh_by_boundingbox(verts,faces,obj_box):
    #PC........

    # 1. 过滤顶点，保留在 obj_box 内的顶点
    mask = (verts >= obj_box[0]) & (verts <= obj_box[1])
    mask = mask.all(dim=1)  # 获取每个顶点的过滤掩码

    # 2. 保留在包围盒内的顶点
    filtered_verts = verts[mask]

    # 3. 调整面矩阵
    # 获取过滤后顶点的索引
    valid_indices = torch.nonzero(mask).squeeze()  # 找到保留的顶点索引
    index_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(valid_indices)}

    # 创建新的面矩阵，只保留所有顶点都在 filtered_verts 内的面
    valid_faces_mask = mask[faces].all(dim=1)  # 只保留所有顶点都在盒子内的面
    filtered_faces = faces[valid_faces_mask]

    # 4. 更新面索引
    # 将过滤后的面矩阵重新映射到新的顶点索引
    for i in range(filtered_faces.shape[0]):
        for j in range(3):
            filtered_faces[i, j] = index_map[filtered_faces[i, j].item()]

    # 最终得到过滤后的顶点和面
    return filtered_verts, filtered_faces


def mesh_texture_transfer(dense_points,dense_faces,gaussian_object,
                          semantic_features,scale_conditioned_semantic_features):
    def SH2RGB(sh):
        C0 = 0.28209479177387814
        return sh * C0 + 0.5
    def RGB2SH(rgb):
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0
    def prepare_scaling_rot(triangles):
        def this_dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            将向量 v 投影到 u 所在的子空间，u 假设已归一化
            """
            coef = this_dot(v, u)
            return coef * u
        """
        近似协方差矩阵并计算缩放和旋转张量，准备参数化的高斯分布
        """
        eps = 1e-8
        v1 = triangles[:, 0].clone()
        v2 = triangles[:, 1].clone()
        v3 = triangles[:, 2].clone()
        v1 = (1/3)*(v1+v2+v3)

        _s2 = v2 - v1
        _s3 = v3 - v1
        
        # 计算法向量 r1
        r1 = torch.cross(_s2, _s3)
        r1 = r1 / (torch.linalg.vector_norm(r1, dim=-1, keepdim=True) + eps)
        # 计算第二个基向量 r2
        s2 = torch.linalg.vector_norm(_s2, dim=-1, keepdim=True) + eps
        r2 = _s2 / s2
        # 计算第三个基向量 r3
        r3 = _s3 - proj(_s3, r1) - proj(_s3, r2)
        r3 = r3 / (torch.linalg.vector_norm(r3, dim=-1, keepdim=True) + eps)

        # 计算 s3
        s3 = this_dot(_s3, r3)
        # 生成缩放张量
        scales = torch.cat([torch.ones_like(s2) * eps,s2, s3], dim=1)
        gaussian_scaling = scales #torch.log(scales)

        # 生成旋转矩阵
        rotation = torch.stack([r1, r2, r3], dim=1)
        rotation = rotation.transpose(-2, -1)  # 转置以得到正确的旋转矩阵

        # 将旋转矩阵转换为四元数
        gaussian_rotation = matrix_to_quaternion(rotation)
        return v1,r1,gaussian_scaling,gaussian_rotation
    
    triangles = dense_points[dense_faces]
    gaus_center,gaus_normals,gaus_scales,gaus_rots = prepare_scaling_rot(triangles)
    
    points,opacities,features,features_extra = gaussian_object 
    print('features.shape: ',features.shape)
    print('features_extra: ',features_extra.shape)
    features_dc = features[:, :1, :].reshape(len(points),-1)
    
    K = 5  # 最近邻的数量
    knn_result = knn_points(torch.tensor(gaus_center).unsqueeze(0).to(torch.float32), points.unsqueeze(0), K=K)
    # 提取最近邻的索引和距离
    idx_nearest = knn_result.idx.squeeze(0)  # 最近邻索引, shape = (M, K)
    dists_nearest = knn_result.dists.squeeze(0)  # 最近邻距离, shape = (M, K)
    # Step 2: 使用反距离加权插值 (IDW) 计算权重
    weights = 1.0 / (dists_nearest + 1e-8)  # 计算权重 (避免除零问题)
    weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化权重, shape = (M, K)
    # 定义一个插值函数，用于对任意特征进行加权平均
    def interpolate_features(features, idx_nearest, weights):
        nearest_features = features[idx_nearest]  # 获取最近邻的特征, shape = (M, K, F)
        interpolated_features = (nearest_features * weights.unsqueeze(-1)).sum(dim=1)  # shape = (M, F)
        return interpolated_features
    
    new_semantic_features = interpolate_features(semantic_features, idx_nearest, weights)
    new_scale_semantic = interpolate_features(scale_conditioned_semantic_features, idx_nearest, weights)

    gaussian_opacities = interpolate_features(opacities, idx_nearest, weights)  # (M, 1)
    gaussian_opacities[gaussian_opacities<0.9] = 0.9
    gaus_features_dc = interpolate_features(features_dc, idx_nearest, weights)[:,None,:] #.view(len(dense_points),3,1)  # (M, F)
    gaus_features_rest = torch.zeros((len(gaus_center),3,15)).cuda().transpose(2,1)
    gaus_features = torch.cat([gaus_features_dc,gaus_features_rest],dim=1)
    gaus_features_extra = interpolate_features(features_extra, idx_nearest, weights)
    return (gaus_center.to(torch.float32),gaus_normals.to(torch.float32),gaus_scales.to(torch.float32),gaus_rots.to(torch.float32),gaussian_opacities.to(torch.float32),
            gaus_features.to(torch.float32),gaus_features_extra.to(torch.float32)),new_semantic_features,new_scale_semantic




def repair_plane_gaussian(gaussian_model,scene_box,obj_box,
                          semantic_features,scale_conditioned_semantic_features):
    import pymeshlab as ml
    centers = gaussian_model.get_xyz

    scene_bounding_box_mask = ((centers>=scene_box[0])&(centers<=scene_box[1])).all(dim=1)
    object_bounding_box_mask = ((centers>=obj_box[0])&(centers<=obj_box[1])).all(dim=1)
    total_mask = scene_bounding_box_mask & (~object_bounding_box_mask)
    centers = gaussian_model.get_xyz[total_mask]
    normals = gaussian_model.get_normals[total_mask]
    opacities = gaussian_model.get_opacity[total_mask]
    features = gaussian_model.get_features[total_mask]
    features_extra = gaussian_model.get_features_extra[total_mask]
    semantic_features = (semantic_features.cuda())[total_mask]
    scale_conditioned_semantic_features = (scale_conditioned_semantic_features.cuda())[total_mask]

    ms = ml.MeshSet()
    #ms.add_mesh(ml.Mesh(vertex_matrix=xyz.astype(np.float64),v_normals_matrix=nms.astype(np.float64)))
    ms.add_mesh(ml.Mesh(vertex_matrix=centers.detach().cpu().numpy(),v_normals_matrix=normals.detach().cpu().numpy()))
    ms.apply_filter('generate_surface_reconstruction_screened_poisson')
    ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=ml.PercentageValue(0.1))
    # extract mesh
    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0
    m = ms.current_mesh()
    verts = torch.tensor(m.vertex_matrix())
    faces = torch.tensor(m.face_matrix())
    verts,faces = crop_mesh_by_boundingbox(verts,faces,obj_box.cpu())
    print('verts.shape: ',verts.shape)
    gaussian_object  = centers,opacities,features,features_extra
    new_gaussians,new_semantic_features,new_scale_semantic = mesh_texture_transfer(verts.cuda(),faces.cuda(),gaussian_object,
                                          semantic_features,scale_conditioned_semantic_features)
    return new_gaussians,new_semantic_features,new_scale_semantic