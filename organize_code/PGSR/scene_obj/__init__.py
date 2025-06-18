#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene_obj.dataset_readers import sceneLoadTypeCallbacks
from scene_obj.gaussian_model import GaussianModel
from arguments import ModelParams
from scene_obj.cameras import Camera
from utils.camera_utils_obj import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self,obj_name, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.obj_name  = obj_name


        self.loaded_iter = None
        self.gaussians = gaussians
        self.source_path = args.source_path

        self.loaded_iter = 30000
        print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
       
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f"cameras_extent {self.cameras_extent}")

        self.multi_view_num = args.multi_view_num # multi-view number 8
        for resolution_scale in resolution_scales: # resolution_scale is [1.0]
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            print("computing nearest_id")
            self.world_view_transforms = []
            for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                self.world_view_transforms.append(cur_cam.world_view_transform)
            self.world_view_transforms = torch.stack(self.world_view_transforms)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "dense_poisson_scripts",
                                                           self.obj_name+"_results",
                                                           "dense_obj_gaussian_0.ply"))
    def save(self, iteration, mask=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud_obj/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), mask)
        print("Saved Object Gaussian to {}".format(os.path.join(point_cloud_path, "point_cloud.ply")))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]