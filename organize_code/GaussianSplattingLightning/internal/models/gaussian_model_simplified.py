import torch
from torch import nn
import internal.utils.gaussian_utils as gaussian_utils
from plyfile import PlyData, PlyElement
from internal.utils.general_utils import inverse_sigmoid
import os
import numpy as np
from internal.utils.sh_utils import eval_sh

class GaussianModelSimplified(nn.Module):
    def __init__(
            self,
            xyz: torch.Tensor,
            normals:torch.Tensor,
            features_dc: torch.Tensor,
            features_rest: torch.Tensor,
            scaling: torch.Tensor,
            rotation: torch.Tensor,
            opacity: torch.Tensor,
            features_extra: torch.Tensor,
            sh_degree: int,
            device,
    ) -> None:
        super().__init__()
        self._xyz = xyz.to(device)
        self._normals = normals.to(device)
        # self._features_dc = features_dc
        # self._features_rest = features_rest
        self._scaling = torch.exp(scaling).to(device)
        self._rotation = torch.nn.functional.normalize(rotation).to(device)
        self._opacity = torch.sigmoid(opacity).to(device)

        # TODO: load only specific dimensions correspond to the sh_degree
        self._features = torch.cat([features_dc, features_rest], dim=1).to(device)

        self._opacity_origin = None

        self._features_extra = features_extra.to(device)

        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree

    def to_device(self, device):
        self._xyz = self._xyz.to(device)
        self._normals = self._normals.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        self._features = self._features.to(device)
        self._features_extra = self._features_extra.to(device)
        return self

    def to(self, *args, **kwargs):
        self._xyz = self._xyz.to(*args, **kwargs)
        self._normals = self._normals.to(*args,**kwargs)
        self._scaling = self._scaling.to(*args, **kwargs)
        self._rotation = self._rotation.to(*args, **kwargs)
        self._opacity = self._opacity.to(*args, **kwargs)
        self._features = self._features.to(*args, **kwargs)
        self._features_extra = self._features_extra.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @classmethod
    def construct_from_state_dict(cls, state_dict, active_sh_degree, device):
        # init_args = {
        #     "sh_degree": active_sh_degree,
        #     "device": device,
        # }
        # for i in state_dict:
        #     if i.startswith("gaussian_model._") is False:
        #         continue
        #     init_args[i[len("gaussian_model._"):]] = state_dict[i]
        #
        # if "features_extra" not in init_args:
        #     init_args["features_extra"] = torch.empty((init_args["xyz"].shape[0], 0))

        gaussian = gaussian_utils.Gaussian.load_from_state_dict(active_sh_degree, state_dict)
        return cls(
            xyz=gaussian.xyz,
            normals=gaussian.normals,
            features_dc=gaussian.features_dc,
            features_rest=gaussian.features_rest,
            scaling=gaussian.scales,
            rotation=gaussian.rotations,
            opacity=gaussian.opacities,
            features_extra=gaussian.real_features_extra,
            sh_degree=active_sh_degree,
            device=device,
        )

    @classmethod
    def construct_from_ply(cls, ply_path: str, sh_degree, device):
        gaussians = gaussian_utils.Gaussian.load_from_ply(ply_path, sh_degree).to_parameter_structure()
        return cls(
            sh_degree=sh_degree,
            device=device,
            xyz=gaussians.xyz,
            normals=gaussians.normals,
            opacity=gaussians.opacities,
            features_dc=gaussians.features_dc,
            features_rest=gaussians.features_rest,
            scaling=gaussians.scales,
            rotation=gaussians.rotations,
            features_extra=gaussians.real_features_extra,
        )

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

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
    def get_opacity(self):
        return self._opacity
    @property
    def get_color(self):
        features_dc = self._features[:, :1, :]
        f_dc = features_dc.reshape((features_dc.shape[0],-1))
        rgbs = np.clip((eval_sh(0, f_dc.unsqueeze(-1), None) + 0.5), 0., 1.)
        rgbs = (rgbs.numpy() * 255).astype(np.uint8)
        return rgbs
    @property
    def get_features_extra(self):
        return self._features_extra

    def select(self, mask: torch.tensor):
        if self._opacity_origin is None:
            self._opacity_origin = torch.clone(self._opacity)  # make a backup
        else:
            self._opacity = torch.clone(self._opacity_origin)

        self._opacity[mask] = 0.
    def axis_ratio(self):
        max_axis = self._scaling.max(dim=1)[0]
        min_axis = self._scaling.min(dim=1)[0]
    def save_mask_gaussians(self,mask,path,with_colors=True):
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
        attribute_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if with_colors is True:
            rgbs = np.clip((eval_sh(0, f_dc.unsqueeze(-1), None) + 0.5), 0., 1.)
            rgbs = (rgbs.numpy() * 255).astype(np.uint8)
            dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            attribute_list.append(rgbs)
    
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attribute_list, axis=1)
        # do not save 'features_extra' for ply
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_extra), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def delete_gaussians(self, mask: torch.tensor):
        gaussians_to_be_preserved = torch.bitwise_not(mask).to(self._xyz.device)

        self._xyz = self._xyz[gaussians_to_be_preserved]
        self._normals = self._normals[gaussians_to_be_preserved]
        self._scaling = self._scaling[gaussians_to_be_preserved]
        self._rotation = self._rotation[gaussians_to_be_preserved]
        #print('self.scaling_: ',self._scaling.shape)
        #print('._rotation: ',self._scaling.shape)
        if self._opacity_origin is not None:
            self._opacity = self._opacity_origin
            self._opacity_origin = None
        self._opacity = self._opacity[gaussians_to_be_preserved]

        self._features = self._features[gaussians_to_be_preserved]
        self._features_extra = self._features_extra[gaussians_to_be_preserved]
    def add_gaussians(self, new_gaussian):
        new_gas_center,new_gas_normals,\
        new_gas_scales,new_gas_rots,\
        new_gas_opac,new_gaus_features,\
        new_gaus_features_extra = new_gaussian
        device = self._xyz.device
        print('new_gas_center.max: ',new_gas_center.max())
        print('self_xyz.max: ',self._xyz.max())
        self._xyz = torch.cat([self._xyz,new_gas_center.to(device)],dim=0)

        print('new_gas_center.max: ',new_gas_normals.max())
        print('self_normals.max: ',self._normals.max())        
        self._normals = torch.cat([self._normals,new_gas_normals.to(device)],dim=0)


        print('new_gas_scaling.max: ',new_gas_scales.max())
        print('self._scaling.max: ',self._scaling.max())
        self._scaling = torch.cat([self._scaling,new_gas_scales.to(device)],dim=0)

        print('new_gas_rotation.max: ',new_gas_rots.max())
        print('self._rotation.max: ',self._rotation.max())
        self._rotation = torch.cat([self._rotation,new_gas_rots.to(device)],dim=0)


        print('new_gas_opacity.max: ',new_gas_opac.max())
        print('self.opacity.max: ',self._opacity.max())
        self._opacity = torch.cat([self._opacity,new_gas_opac.to(device)],dim=0)

        print('new_gas_features.max: ',new_gaus_features.max())
        print('self._features.max: ',self._features.max())
        self._features = torch.cat([self._features,new_gaus_features.to(device)],dim=0)
        self._features_extra = torch.cat([self._features_extra,new_gaus_features_extra.to(device)],dim=0)
    def to_parameter_structure(self) -> gaussian_utils.Gaussian:
        xyz = self._xyz.cpu()
        normals = self._normals.cpu()
        features_dc = self._features[:, :1, :].cpu()
        features_rest = self._features[:, 1:, :].cpu()
        scaling = torch.log(self._scaling).cpu()
        rotation = self._rotation.cpu()
        opacity = inverse_sigmoid(self._opacity).cpu()
        features_extra = self._features_extra.cpu()

        return gaussian_utils.Gaussian(
            sh_degrees=self.max_sh_degree,
            xyz=xyz,
            normals=normals,
            opacities=opacity,
            features_dc=features_dc,
            features_rest=features_rest,
            scales=scaling,
            rotations=rotation,
            real_features_extra=features_extra,
        )

    def to_ply_structure(self) -> gaussian_utils.Gaussian:
        xyz = self._xyz.cpu().numpy()
        normals = self._normals.cpu().numpy()
        features_dc = self._features[:, :1, :].transpose(1, 2).cpu().numpy()
        features_rest = self._features[:, 1:, :].transpose(1, 2).cpu().numpy()
        scaling = torch.log(self._scaling).cpu().numpy()
        rotation = self._rotation.cpu().numpy()
        opacity = inverse_sigmoid(self._opacity).cpu().numpy()
        features_extra = self._features_extra.cpu().numpy()

        return gaussian_utils.Gaussian(
            sh_degrees=self.max_sh_degree,
            xyz=xyz,
            normals=normals,
            opacities=opacity,
            features_dc=features_dc,
            features_rest=features_rest,
            scales=scaling,
            rotations=rotation,
            real_features_extra=features_extra,
        )
