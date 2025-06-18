import torch
import numpy as np
import internal.utils.gaussian_utils as gaussian_utils
from .gaussian_model_simplified import GaussianModelSimplified
from internal.utils.general_utils import inverse_sigmoid
from plyfile import PlyData, PlyElement
import os
from internal.utils.sh_utils import eval_sh
from pytorch3d.transforms import quaternion_to_matrix
class SimplifiedGaussianModelManager:
    models: list = None

    # setup methods
    select = GaussianModelSimplified.select
    _delete_gaussians = GaussianModelSimplified.delete_gaussians
    _add_gaussians = GaussianModelSimplified.add_gaussians
    to_parameter_structure = GaussianModelSimplified.to_parameter_structure
    to_ply_structure = GaussianModelSimplified.to_ply_structure

    def __init__(self, simplified_gaussian_models: list[GaussianModelSimplified], enable_transform: bool, device):
        super().__init__()
        if enable_transform is True:
            self.models = simplified_gaussian_models
        self.device = device

        # calculate total gaussian num
        total_gaussian_num = 0
        model_gaussian_indices = []
        for i in simplified_gaussian_models:
            n = i.get_xyz.shape[0]
            model_gaussian_indices.append((total_gaussian_num, total_gaussian_num + n))
            total_gaussian_num += n

        self.model_gaussian_indices = model_gaussian_indices
        # create tensor
        tensor_initialize_params = {
            "dtype": torch.float,
            "device": device,
        }
        self._xyz = torch.zeros((total_gaussian_num, 3), **tensor_initialize_params)
        self._normals = torch.zeros((total_gaussian_num, 3), **tensor_initialize_params)
        self._opacity = torch.zeros((total_gaussian_num, 1), **tensor_initialize_params)
        self._features = torch.zeros([total_gaussian_num] + list(simplified_gaussian_models[0].get_features.shape[1:]), **tensor_initialize_params)
        self._scaling = torch.zeros((total_gaussian_num, 3), **tensor_initialize_params)
        self._rotation = torch.zeros((total_gaussian_num, 4), **tensor_initialize_params)
        self._features_extra = torch.zeros((total_gaussian_num, simplified_gaussian_models[0].get_features_extra.shape[-1]), **tensor_initialize_params)

        # merge gaussians
        for idx, model in enumerate(simplified_gaussian_models):
            begin, end = self.model_gaussian_indices[idx]
            #if idx==4:
            #    self._xyz[begin:end] = model.get_xyz.to(device)+torch.tensor([-0.2,-0.5,-0.2]).cuda()
            #else:
            self._xyz[begin:end] = model.get_xyz.to(device)
            self._normals[begin:end] = model.get_normals.to(device)
            self._opacity[begin:end] = model.get_opacity.to(device)
            self._features[begin:end] = model.get_features.to(device)
            self._scaling[begin:end][:, :model.get_scaling.shape[-1]] = model.get_scaling.to(device)
            self._rotation[begin:end] = model.get_rotation.to(device)
            self._features_extra[begin:end] = model.get_features_extra.to(device)

        self.max_sh_degree = simplified_gaussian_models[0].max_sh_degree
        self.active_sh_degree = simplified_gaussian_models[0].max_sh_degree

        self._opacity_origin = None

        self.mask = (self._opacity>0.00)[:,0] # pls do not modify that, otherwise bug
        for idx, model in enumerate(self.models):
            begin, end = self.model_gaussian_indices[idx]
            model.delete_gaussians((torch.bitwise_not(self.mask))[begin:end])
        self._delete_gaussians(torch.bitwise_not(self.mask))
        """
        self._xyz=self._xyz[self.mask,:]
        self._normals=self._normals[self.mask,:]
        self._opacity = self._opacity[self.mask,:]
        self._features = self._features[self.mask,:]
        self._scaling = self._scaling[self.mask,:]
        self._rotation = self._rotation[self.mask,:]
        self._features_extra = self._features_extra[self.mask,:]
        """
    def get_model_gaussian_indices(self, idx: int):
        return self.model_gaussian_indices[idx]

    def get_model(self, idx: int) -> GaussianModelSimplified:
        return self.models[idx]

    def transform_with_vectors(
            self,
            idx: int,
            scale: float,
            r_wxyz: np.ndarray,
            t_xyz: np.ndarray,
    ):
        model = self.get_model(idx)
        begin, end = self.get_model_gaussian_indices(idx)

        xyz = model.get_xyz.to(self.device)
        normals = model.get_normals.to(self.device)
        # TODO: avoid memory copy if no rotation or scaling happened compared to previous state
        scaling = model.get_scaling.to(self.device)
        rotation = model.get_rotation.to(self.device)
        features = model.get_features.to(self.device)  # consume a lot of memory

        # rescale
        xyz, scaling = gaussian_utils.GaussianTransformUtils.rescale(
            xyz,
            scaling,
            scale
        )
        normals, _, _ = gaussian_utils.GaussianTransformUtils.rotate_by_wxyz_quaternions(
            xyz=normals,
            rotations=rotation,
            features=features,
            quaternions=torch.tensor(r_wxyz).to(normals),
        )
        # rotate
        xyz, rotation, new_features = gaussian_utils.GaussianTransformUtils.rotate_by_wxyz_quaternions(
            xyz=xyz,
            rotations=rotation,
            features=features,
            quaternions=torch.tensor(r_wxyz).to(xyz),
        )
        # translate
        xyz = gaussian_utils.GaussianTransformUtils.translation(xyz, *t_xyz.tolist())
        self._normals[begin:end] = normals
        self._xyz[begin:end] = xyz
        self._scaling[begin:end] = scaling
        self._rotation[begin:end] = rotation
        self._features[begin:end] = new_features

    def transform(
            self,
            idx: int,
            scale: float,
            rx: float,
            ry: float,
            rz: float,
            tx: float,
            ty: float,
            tz: float,
    ):
        model = self.get_model(idx)
        begin, end = self.get_model_gaussian_indices(idx)

        xyz = model.get_xyz.to(self.device)
        # TODO: avoid memory copy if no rotation or scaling happened compared to previous state
        scaling = model.get_scaling.to(self.device)
        rotation = model.get_rotation.to(self.device)

        xyz, scaling = gaussian_utils.GaussianTransformUtils.rescale(
            xyz,
            scaling,
            scale
        )
        xyz, rotation = gaussian_utils.GaussianTransformUtils.rotate_by_euler_angles(
            xyz,
            rotation,
            rx,
            ry,
            rz,
        )
        normals, rotation = gaussian_utils.GaussianTransformUtils.rotate_by_euler_angles(
            normals,
            rotation,
            rx,
            ry,
            rz,
        )
        xyz = gaussian_utils.GaussianTransformUtils.translation(
            xyz,
            tx,
            ty,
            tz,
        )

        self._xyz[begin:end] = xyz
        self._normals[begin:end]= normals
        self._scaling[begin:end] = scaling
        self._rotation[begin:end] = rotation

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    def get_normal(self, camera_center):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    @property
    def get_xyz(self):
        return self._xyz
    @property
    def get_normals(self):
        return self._normals
    @property
    def get_features(self):
        return self._features
    @property
    def get_color(self):
        features_dc = self._features[:, :1, :]
        f_dc = features_dc.reshape((features_dc.shape[0],-1)).cpu()
        rgbs = np.clip((eval_sh(0, f_dc.unsqueeze(-1), None) + 0.5), 0., 1.)
        #rgbs = (rgbs.numpy() * 255).astype(np.uint8)
        return rgbs
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_features_extra(self):
        return self._features_extra
    def axis_ratio(self):
        max_axis = self._scaling.max(dim=1)[0]
        min_axis = self._scaling.min(dim=1)[0]
        return max_axis/min_axis
    
    def add_gaussians(self, new_gaussians):
        # delete form each model, build new indices
        total_gaussian_num = 0
        model_gaussian_indices = []
        for idx, model in enumerate(self.models):
            begin, end = self.model_gaussian_indices[idx]
            model.add_gaussians(new_gaussians)

            # build new indices
            n = model.get_xyz.shape[0]
            model_gaussian_indices.append((total_gaussian_num, total_gaussian_num + n))
            total_gaussian_num += n
        # update indices
        self.model_gaussian_indices = model_gaussian_indices

        self._add_gaussians(new_gaussians)

        assert self._xyz.shape[0] == total_gaussian_num
    def delete_gaussians(self, mask: torch.tensor):
        # delete form each model, build new indices
        total_gaussian_num = 0
        model_gaussian_indices = []
        for idx, model in enumerate(self.models):
            begin, end = self.model_gaussian_indices[idx]
            model.delete_gaussians(mask[begin:end])

            # build new indices
            n = model.get_xyz.shape[0]
            model_gaussian_indices.append((total_gaussian_num, total_gaussian_num + n))
            total_gaussian_num += n
        # update indices
        self.model_gaussian_indices = model_gaussian_indices

        self._delete_gaussians(mask)

        assert self._xyz.shape[0] == total_gaussian_num
    
    def save_mask_gaussians(self, mask,path,with_colors=True):
        """
        scaling = torch.log(self._scaling).cpu().numpy()
        rotation = self._rotation.cpu().numpy()
        opacity = inverse_sigmoid(self._opacity).cpu().numpy()
        """
        
        mask = mask.to(self._xyz.device)
        xyz = self._xyz[mask]
        normals = self._normals[mask]
        scale = torch.log(self._scaling[mask])
        rotation = self._rotation[mask]
        opacities = inverse_sigmoid(self._opacity[mask])
        
        features_dc = self._features[:, :1, :]
        features_rest = self._features[:, 1:, :]
        f_dc = features_dc[mask]
        f_dc = f_dc.reshape((f_dc.shape[0],-1))
        
        f_rest = features_rest[mask]
        f_rest = f_rest.reshape((f_rest.shape[0],-1))
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # f_extra = gaussian.real_features_extra

        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the 3 DC
            for i in range(features_dc.shape[1] * features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            if self.max_sh_degree > 0:
                for i in range(features_rest.shape[1] * features_rest.shape[2]):
                    l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(scale.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(rotation.shape[1]):
                l.append('rot_{}'.format(i))
            # for i in range(self.real_features_extra.shape[1]):
            #     l.append('f_extra_{}'.format(i))
            return l

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
        attribute_list = [xyz.cpu(), normals.cpu(), f_dc.cpu(), f_rest.cpu(), opacities.cpu(), scale.cpu(), rotation.cpu()]
        if with_colors is True:
            from internal.utils.sh_utils import eval_sh
            print('f_dc.shape: ',f_dc.shape)
            rgbs = np.clip((eval_sh(0, f_dc.cpu().unsqueeze(-1), None) + 0.5), 0., 1.)
            print('this_rgb.shape: ',rgbs.shape)
            rgbs = (rgbs.numpy() * 255).astype(np.uint8)
            print('rgbs.shape: ',rgbs.shape)
            dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            attribute_list.append(rgbs)
    
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attribute_list, axis=1)
        # do not save 'features_extra' for ply
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_extra), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
