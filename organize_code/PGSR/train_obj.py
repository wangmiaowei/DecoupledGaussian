#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
from datetime import datetime
import torch
import random
import numpy as np
from PIL import Image
from pytorch3d.ops import knn_points
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp
from scene_obj.gaussian_model import GaussianModel
from scene_obj.app_model import AppModel
from gaussian_renderer import render, network_gui
import sys, time

from torchvision.utils import save_image
from scene_obj import Scene, GaussianModel
from utils.general_utils import safe_state
import torchvision.models as models
import uuid
from tqdm import tqdm
from utils.general_utils import inverse_sigmoid
from utils.image_utils import psnr, erode,resize,cubic,cosine_avg
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene_obj.app_model import AppModel
from scene_obj.cameras import Camera
import torchvision.models as models
import torchvision.transforms as transforms

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def training(fine_step,obj_name,dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    opt.iterations = 30_000 + fine_step
    saving_iterations =[opt.iterations]
    # configure path
    dataset.source_path = os.path.join("../input_dataset", obj_name)
    dataset.model_path = os.path.join("../exp_res", obj_name)
    first_iter = checkpoint_iterations[0]
    
    prepare_output_and_logger(dataset)

    
    gaussians = GaussianModel(dataset.sh_degree)

       
     

    scene = Scene(obj_name,dataset, gaussians,load_iteration=checkpoint_iterations[0])
    gaussians.training_setup(opt)
    app_model = AppModel()
    app_model.train()
    app_model.cuda()



    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    def binary_cross_entropy(y_pred, y_true):
        # Avoid the issue of numerical instability caused by log(0)
        epsilon = 1e-8
        # Limit the predicted value to be within the range of (epsilon, 1 - epsilon)
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

        # Calculate the BCE loss
        loss = - (0.01*y_true * torch.log(y_pred) + 0.99*(1 - y_true) * torch.log(1 - y_pred))

        # Return the average loss
        return loss.mean()


    gaussians.update_learning_rate(30000)
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            # To print the learning rate
            print("gaussian_xyz:", gaussians._xyz[10])
            print("gaussian_featuresdc:", gaussians._features_dc[10])
            print("gaussian_featuresrest:", gaussians._features_rest[10])
            print("gaussian_scaling:", gaussians._scaling[10])
            print("gaussian_rotation:", gaussians._rotation[10])
            print("gaussian_opacities:", gaussians._opacity[10])
            for param_group in gaussians.optimizer.param_groups:
                print(f"Learning rate: {param_group['lr']}")
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam =viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image,mask_image,_ = viewpoint_cam.get_image()
        mask_image[mask_image>0]=1
        if torch.count_nonzero(mask_image)<40:
            continue
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") #if opt.random_background else background
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter)
        

        image,mask_alpha,visibility_filter = render_pkg["render"], render_pkg["mask"],render_pkg["visibility_filter"]

        silhouette_loss = binary_cross_entropy(mask_alpha.float(),mask_image.float())
        gt_mask_images = gt_image*mask_image
        gt_mask_images[:,mask_image.squeeze(0)==0] = bg[:,None]

        """
        # Debugging: Save images to check the output
        save_image(image, 'image.png')
        save_image(gt_mask_images, 'gt_mask.png')
        """
        ssim_loss = (1.0 - ssim(image, gt_mask_images))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_mask_images)
        else:
            Ll1 = l1_loss(image, gt_mask_images)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()+ 1e-4*silhouette_loss.clone()


        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            max_scale = sorted_scale[..., -1]
            threshold = 3  # Adjust as neede
            scale_ratio = max_scale / sorted_scale[...,0]
            penalty = torch.clamp(scale_ratio - threshold, min=0.0)
            scale_ratio_loss = penalty.mean()
            loss += 10.0 * scale_ratio_loss  # Adjust the coefficient as 

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = loss.item()#0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    #"Silhouette_loss": f"{silhouette_loss.item():.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)

            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)#test_hhaa

            gaussians.optimizer.step()
            app_model.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            app_model.optimizer.zero_grad(set_to_none = True)

            if iteration % 500 == 0:
                torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    print("\n[ITER {}] Saving Gaussians".format(opt.iterations))
    scene.save(opt.iterations)#test_hhaa

    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('obj_name', type=str, help='Name of the object')
    parser.add_argument('--finetune_iter', type=int, default=7000)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    

    # Initialize system state (RNG)
    safe_state(args.quiet)
    fine_step = args.finetune_iter
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(fine_step,args.obj_name,lp.extract(args), 
             op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
