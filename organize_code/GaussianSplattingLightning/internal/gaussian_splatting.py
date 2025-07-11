import os.path
import queue
import threading
import traceback
from typing import Tuple, List, Dict, Union, Any, Callable, Optional
from typing_extensions import Self

import torch.optim
import torchvision
import wandb
import csv
from lightning.pytorch.core.module import MODULE_OPTIMIZERS
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, LRSchedulerPLType, STEP_OUTPUT
import lightning.pytorch.loggers

import internal.mp_strategy
from internal.viewer.training_viewer import TrainingViewer
from internal.configs.model import ModelParams
from internal.configs.light_gaussian import LightGaussian

from internal.models.gaussian_model import GaussianModel
from internal.renderers import Renderer, VanillaRenderer, RendererConfig
from internal.metrics.metric import Metric
from internal.metrics.vanilla_metrics import VanillaMetrics
from internal.density_controllers.density_controller import DensityController
from internal.density_controllers.vanilla_density_controller import VanillaDensityController
from jsonargparse import lazy_instance

from internal.utils.sh_utils import eval_sh
from internal.utils.graphics_utils import store_ply


class GaussianSplatting(LightningModule):
    def __init__(
            self,
            gaussian: ModelParams,
            light_gaussian: LightGaussian,
            save_iterations: List[int],
            background_color: Tuple[float, float, float] = (0., 0., 0.),
            random_background: bool = False,
            output_path: str = None,
            save_val_output: bool = False,
            save_val_metrics: bool = None,
            max_save_val_output: int = -1,
            renderer: Union[Renderer, RendererConfig] = lazy_instance(VanillaRenderer),
            metric: Metric = lazy_instance(VanillaMetrics),
            density: DensityController = lazy_instance(VanillaDensityController),
            save_ply: bool = False,
            web_viewer: bool = False,
            initialize_from: str = None,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # setup models
        self.gaussian_model = GaussianModel(sh_degree=gaussian.sh_degree, extra_feature_dims=gaussian.extra_feature_dims)

        self.optimization_hparams = self.hparams["gaussian"].optimization
        self.light_gaussian_hparams = light_gaussian

        # instantiate renderer
        if isinstance(renderer, RendererConfig):
            renderer = renderer.instantiate()
        self.renderer = renderer

        # instantiate density controller
        self.density_controller = density.instantiate()

        # metrics
        self.metric = metric.instantiate()

        # background color
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        if random_background is True:
            self.get_background_color = self._random_background_color
        else:
            self.get_background_color = self._fixed_background_color

        self.web_viewer: TrainingViewer = None

        self.batch_size = 1
        self.restored_epoch = 0
        self.restored_global_step = 0

        self.max_image_saving_threads = 16
        self.image_queue = queue.Queue(maxsize=self.max_image_saving_threads)
        self.image_saving_threads = []

        self.val_metrics: List[Tuple[str, Dict]] = []

        # hooks
        self.on_train_batch_end_hooks: List[Callable[[Dict, Any, GaussianModel, int, Self], None]] = []

    def log_metrics(
            self,
            metrics: dict,
            prog_bar: dict,
            prefix: str,
            on_step: bool,
            on_epoch: bool,
            name_prefix: str = "",
    ):
        for name in metrics:
            self.log(
                f"{prefix}/{name_prefix}{name}",
                metrics[name],
                prog_bar=prog_bar[name],
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=self.batch_size,
            )

    def _fixed_background_color(self):
        return self.background_color

    def _random_background_color(self):
        return torch.rand(3)

    def _initialize_gaussians_from_trained_model(self):
        assert self.hparams["gaussian"].extra_feature_dims == 0

        from internal.utils.gaussian_model_loader import GaussianModelLoader
        load_from = GaussianModelLoader.search_load_file(self.hparams["initialize_from"])
        if load_from.endswith(".ply") is True:
            self.gaussian_model.load_ply(load_from, self.device)
        else:
            # load from ckpt
            ckpt = torch.load(load_from, map_location=self.device)
            # update model hyper params
            self.gaussian_model.extra_feature_dims = ckpt["hyper_parameters"]["gaussian"].extra_feature_dims
            self.gaussian_model.max_sh_degree = ckpt["hyper_parameters"]["gaussian"].sh_degree
            self.gaussian_model.active_sh_degree = self.gaussian_model.max_sh_degree
            # initialize params
            self.gaussian_model.initialize_by_gaussian_number(ckpt["state_dict"]["gaussian_model._xyz"].shape[0])
            # load state_dict
            state_dict = {}
            for i in ckpt["state_dict"]:
                if i.startswith("gaussian_model."):
                    state_dict[i[i.find(".") + 1:]] = ckpt["state_dict"][i]
            self.gaussian_model.load_state_dict(state_dict)

        # update hyper params
        self.hparams["gaussian"].extra_feature_dims = self.gaussian_model.extra_feature_dims
        self.hparams["gaussian"].sh_degree = self.gaussian_model.max_sh_degree

        print(f"initialize from {load_from}: sh_degree={self.gaussian_model.max_sh_degree}, extra_feature_dims={self.gaussian_model.extra_feature_dims}")

    def setup(self, stage: str):
        if stage == "fit":
            if self.hparams["initialize_from"] is None:
                self.gaussian_model.create_from_pcd(
                    self.trainer.datamodule.point_cloud,
                    deivce=self.device,
                )
            else:
                self._initialize_gaussians_from_trained_model()
        else:
            if self.hparams["save_val_metrics"] is None:
                self.hparams["save_val_metrics"] = True

        self.renderer.setup(stage=stage, lightning_module=self)
        self.metric.setup(stage=stage, pl_module=self)
        self.density_controller.setup(stage=stage, pl_module=self)

        # use different image log method based on the logger type
        self.log_image = None
        if isinstance(self.logger, lightning.pytorch.loggers.TensorBoardLogger):
            self.log_image = self.tensorboard_log_image
        elif isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.log_image = self.wandb_log_image

    def on_load_checkpoint(self, checkpoint) -> None:
        # reinitialize parameters based on the gaussian number in the checkpoint
        self.gaussian_model.initialize_by_gaussian_number(checkpoint["state_dict"]["gaussian_model._xyz"].shape[0])
        # restore some extra parameters
        if "gaussian_model_extra_state_dict" in checkpoint:
            for i in checkpoint["gaussian_model_extra_state_dict"]:
                setattr(self.gaussian_model, i, checkpoint["gaussian_model_extra_state_dict"][i])
            # for previous version
            if "active_sh_degree" not in checkpoint["gaussian_model_extra_state_dict"]:
                self.gaussian_model.active_sh_degree = self.gaussian_model.max_sh_degree

        if "gaussian_model._features_extra" not in checkpoint["state_dict"]:
            # create an empty `_features_extra`
            checkpoint["state_dict"]["gaussian_model._features_extra"] = torch.zeros_like(self.gaussian_model._features_extra)
            # create an optimizer param group
            new_param_groups = checkpoint["optimizer_states"][0]["param_groups"][0].copy()
            new_param_groups["name"] = "f_extra"
            new_param_groups["lr"] = 0.
            new_param_groups["params"] = [len(checkpoint["optimizer_states"][0]["param_groups"])]
            checkpoint["optimizer_states"][0]["param_groups"].append(new_param_groups)

        # get epoch and global_step, which used in the output path of the validation and test images
        self.restored_epoch = checkpoint["epoch"]
        self.restored_global_step = checkpoint["global_step"]

        # call for renderer
        self.renderer.on_load_checkpoint(self, checkpoint)
        # call density controller's hook
        self.density_controller.on_load_checkpoint(self, checkpoint)

        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint) -> None:
        # store some extra parameters
        checkpoint["gaussian_model_extra_state_dict"] = {
            "max_radii2D": self.gaussian_model.max_radii2D,
            "xyz_gradient_accum": self.gaussian_model.xyz_gradient_accum,
            "denom": self.gaussian_model.denom,
            "spatial_lr_scale": self.gaussian_model.spatial_lr_scale,
            "active_sh_degree": self.gaussian_model.active_sh_degree,
        }
        super().on_save_checkpoint(checkpoint)

    def tensorboard_log_image(self, tag: str, image_tensor):
        self.logger.experiment.add_image(
            tag,
            image_tensor,
            self.trainer.global_step,
        )

    def wandb_log_image(self, tag: str, image_tensor):
        image_dict = {
            tag: wandb.Image(image_tensor),
        }
        self.logger.experiment.log(
            image_dict,
            step=self.trainer.global_step,
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        if batch[0].device != self.device:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        camera, image_info, extra_data = batch
        image_name, gt_image, masked_pixels = image_info

        if extra_data is not None:
            extra_data = super().transfer_batch_to_device(extra_data, device, dataloader_idx)
        if gt_image is not None:
            gt_image = gt_image.to(device)
        if masked_pixels is not None:
            masked_pixels = masked_pixels.to(device)

        return camera, (image_name, gt_image, masked_pixels), extra_data

    def forward(self, camera):
        if self.training is True:
            return self.renderer.training_forward(
                self.trainer.global_step,
                self,
                camera,
                self.gaussian_model,
                bg_color=self.get_background_color().to(camera.R.device),
            )
        return self.renderer(
            camera,
            self.gaussian_model,
            bg_color=self._fixed_background_color().to(camera.R.device),
        )

    def optimizers(self, use_pl_optimizer: bool = True):
        optimizers = super().optimizers(use_pl_optimizer=use_pl_optimizer)

        if isinstance(optimizers, list) is False:
            return [optimizers]

        """
        IMPORTANCE: the global_step will be increased on every step() call of all the optimizers,
        issue https://github.com/Lightning-AI/lightning/issues/17958,
        here change _on_before_step and _on_after_step to override this behavior.
        """
        for idx, optimizer in enumerate(optimizers):
            if idx == 0:
                continue
            optimizer._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
            optimizer._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")

        return optimizers

    def lr_schedulers(self) -> Union[None, List[LRSchedulerPLType], LRSchedulerPLType]:
        schedulers = super().lr_schedulers()

        if schedulers is None:
            return []

        if isinstance(schedulers, list) is False:
            return [schedulers]

        return schedulers

    def is_final_step(self, step: int = None):
        if step is None:
            step = self.trainer.global_step
        if self.trainer.max_steps > 0 and step >= self.trainer.max_steps:
            return True
        # TODO: make it works when max_epochs set
        return False

    def on_train_start(self) -> None:
        super().on_train_start()

        if self.hparams["web_viewer"] is True and self.trainer.global_rank == 0:
            if self.trainer.datamodule.hparams["parser"].__class__.__name__.lower() in ["blender", "nsvf", "matrixcity"]:
                up = torch.tensor([0., 0., 1.])
            else:
                c2w = self.trainer.datamodule.dataparser_outputs.train_set.cameras.world_to_camera[:, :3, :3]
                up = c2w[:, :3, 1].mean(dim=0)
                up = -up / torch.linalg.norm(up)
            self.web_viewer = TrainingViewer(
                camera_names=self.trainer.datamodule.dataparser_outputs.train_set.image_names,
                cameras=self.trainer.datamodule.dataparser_outputs.train_set.cameras,
                up_direction=up.cpu().numpy(),
                camera_center=self.trainer.datamodule.dataparser_outputs.train_set.cameras.camera_center.mean(dim=0).cpu().numpy(),
                available_appearance_options=self.trainer.datamodule.dataparser_outputs.appearance_group_ids,
            )
            self.web_viewer.start()

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        if self.web_viewer is not None:
            self.web_viewer.training_step(
                self.gaussian_model,
                self.renderer,
                self._fixed_background_color(),
                self.trainer.global_step,
            )
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        camera, image_info, _ = batch
        # image_name, gt_image, masked_pixels = image_info

        global_step = self.trainer.global_step + 1  # must start from 1 to prevent densify at the beginning

        # get optimizers and schedulers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()

        # zero grad
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        # save checkpoint
        # checkpoint will always be saved after final step, so do not save for final step here
        if global_step in self.hparams["save_iterations"] and self.is_final_step(global_step) is False and self.trainer.global_step != self.restored_global_step:
            self.save_gaussians()

        # call renderer hook
        self.renderer.before_training_step(global_step, self)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if global_step % 1000 == 0:
            self.gaussian_model.oneupSHdegree()

        # forward
        outputs = self(camera)
        # metrics
        metrics, prog_bar = self.metric.get_train_metrics(self, self.gaussian_model, global_step, batch, outputs)
        self.log_metrics(metrics, prog_bar, prefix="train", on_step=True, on_epoch=False)

        # log learning rate and gaussian count every 100 iterations (without plus one step)
        if self.trainer.global_step % 100 == 0:
            metrics_to_log = {
                "train/gaussians_count": self.gaussian_model.get_xyz.shape[0],
            }
            for opt_idx, opt in enumerate(optimizers):
                if opt is None:
                    continue
                for idx, param_group in enumerate(opt.param_groups):
                    param_group_name = param_group["name"] if "name" in param_group else str(idx)
                    metrics_to_log["lr/{}_{}".format(opt_idx, param_group_name)] = param_group["lr"]
            self.logger.log_metrics(
                metrics_to_log,
                step=self.trainer.global_step,
            )

        # backward
        self.manual_backward(metrics["loss"])

        # invoke densify controller before gradient descend
        self.density_controller(outputs, batch, self.gaussian_model, global_step, self)

        # optimize
        for optimizer in optimizers:
            optimizer.step()

        # schedule lr
        self.gaussian_model.update_learning_rate(global_step)
        for scheduler in schedulers:
            scheduler.step()

    def light_gaussian_prune(self, global_step):
        """
        LightGaussian prune
        """
        if global_step not in self.light_gaussian_hparams.prune_steps:
            return

        from internal.utils.light_gaussian import get_count_and_score
        from internal.utils.light_gaussian import calculate_v_imp_score
        from internal.utils.light_gaussian import get_prune_mask

        # try to detect whether anti aliased enabled
        try:
            anti_aliased = self.renderer.anti_aliased
        except:
            anti_aliased = False

        with torch.no_grad():
            count, score, _, _ = get_count_and_score(
                self.gaussian_model,
                self.trainer.datamodule.dataparser_outputs.train_set.cameras,
                anti_aliased,
            )
            v_list = calculate_v_imp_score(
                self.gaussian_model.get_scaling,
                score,
                self.light_gaussian_hparams.v_pow,
            )

            # TODO: `self.light_gaussian_hparams.prune_steps` should be sorted
            prune_step_index = self.light_gaussian_hparams.prune_steps.index(global_step)
            prune_percent = self.light_gaussian_hparams.prune_percent * (self.light_gaussian_hparams.prune_decay ** prune_step_index)
            prune_mask = get_prune_mask(prune_percent, v_list)

            print(f"number_of_gaussian={self.gaussian_model.get_xyz.shape[0]}, "
                  f"number_to_prune={prune_mask.sum().item()}, "
                  f"prune_percent={prune_percent}, "
                  f"anti_aliased={anti_aliased}")
            self.gaussian_model.prune_points(prune_mask)
            print(f"number_of_gaussian_after_pruning={self.gaussian_model.get_xyz.shape[0]}")

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # the value of `trainer.global_step` here
        # is the same as the local variable `global_step` in training_step
        global_step = self.trainer.global_step

        self.light_gaussian_prune(global_step)

        self.renderer.after_training_step(self.trainer.global_step, self)

        for i in self.on_train_batch_end_hooks:
            i(outputs, batch, self.gaussian_model, global_step, self)

        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        if self.web_viewer is not None:
            self.web_viewer.validation_step(
                self.gaussian_model,
                self.renderer,
                self._fixed_background_color(),
                batch_idx,
            )

    def validation_step(self, batch, batch_idx, name: str = "val"):
        camera, image_info, _ = batch
        gt_image = image_info[1]

        # forward
        outputs = self(camera)
        metrics, prog_bar = self.metric.get_validate_metrics(self, self.gaussian_model, batch, outputs)
        self.log_metrics(metrics, prog_bar, prefix=name, on_step=False, on_epoch=True)
        self.val_metrics.append((image_info[0], metrics))

        # write validation image
        if self.trainer.global_rank == 0 and self.hparams["save_val_output"] is True and (
                self.hparams["max_save_val_output"] < 0 or batch_idx < self.hparams["max_save_val_output"]
        ):
            self.image_queue.put({
                "output_image": outputs["render"].cpu(),
                "extra_image": outputs["extra_image"].cpu() if "extra_image" in outputs else None,
                "gt_image": gt_image.cpu(),
                "stage": name,
                "image_name": image_info[0],
                "epoch": max(self.trainer.current_epoch, self.restored_epoch),
                "step": max(self.trainer.global_step, self.restored_global_step),
            })

            # if self.log_image is not None:
            #     grid = torchvision.utils.make_grid(torch.concat([outputs["render"], gt_image], dim=-1))
            #     self.log_image(
            #         tag="{}_images/{}".format(name, image_info[0].replace("/", "_")),
            #         image_tensor=grid,
            #     )
            #
            # image_output_path = os.path.join(
            #     self.hparams["output_path"],
            #     name,
            #     "epoch={}-step={}".format(
            #         max(self.trainer.current_epoch, self.restored_epoch),
            #         max(self.trainer.global_step, self.restored_global_step),
            #     ),
            #     "{}.png".format(image_info[0].replace("/", "_"))
            # )
            # os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
            # torchvision.utils.save_image(
            #     torch.concat([outputs["render"], gt_image], dim=-1),
            #     image_output_path,
            # )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if self.hparams["save_val_output"] is True:
            for i in range(self.max_image_saving_threads):
                thread = threading.Thread(target=self.save_images)
                self.image_saving_threads.append(thread)
                thread.start()

    def on_validation_epoch_end(self, name="val") -> None:
        super().on_validation_epoch_end()
        for i in range(len(self.image_saving_threads)):
            self.image_queue.put(None)
        for i in self.image_saving_threads:
            i.join()
        self.image_saving_threads = []

        # save metrics
        if self.hparams["save_val_metrics"] is True and self.global_rank == 0 and len(self.val_metrics) > 0:
            metrics_output_dir = os.path.join(self.hparams["output_path"], "metrics")
            os.makedirs(metrics_output_dir, exist_ok=True)
            step = max(self.trainer.global_step, self.restored_global_step)

            metric_list_key_by_name = {}  # [metric_name] = metric_value_list
            metric_fields = list(self.val_metrics[0][1].keys())
            for i in metric_fields:
                metric_list_key_by_name[i] = []

            with open(os.path.join(metrics_output_dir, f"{name}-step={step}.csv"), "w") as f:
                metrics_writer = csv.writer(f)
                metrics_writer.writerow(["name"] + list(metric_fields))

                for image_name, metrics in self.val_metrics:
                    metric_row = [image_name]
                    for metric_name in metric_fields:
                        metric_list_key_by_name[metric_name].append(metrics[metric_name])
                        metric_row.append("{:.8f}".format(metrics[metric_name].item()))
                    metrics_writer.writerow(metric_row)

                # calculate mean metrics
                metrics_writer.writerow([""] + ["" for _ in range(len(metric_fields))])
                mean_metrics = ["MEAN"]
                for i in metric_fields:
                    mean_metrics.append("{:.8f}".format(torch.stack(metric_list_key_by_name[i]).mean(dim=0).item()))
                metrics_writer.writerow(mean_metrics)

        self.val_metrics.clear()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self.on_validation_epoch_end(name="test")

    def save_images(self):
        while True:
            item = self.image_queue.get()
            if item is None:
                break

            try:
                image_list = []
                if item["extra_image"] is not None:
                    extra_image = item["extra_image"]
                    if extra_image.shape[0] == 1:
                        extra_image = extra_image.repeat(3, 1, 1)
                    image_list.append(extra_image)
                image_list += [item["output_image"], item["gt_image"]]
                image = torch.concat(image_list, dim=-1)

                if self.log_image is not None:
                    grid = torchvision.utils.make_grid(image)
                    self.log_image(
                        tag="{}_images/{}".format(item["stage"], item["image_name"].replace("/", "_")),
                        image_tensor=grid,
                    )

                image_output_path = os.path.join(
                    self.hparams["output_path"],
                    item["stage"],
                    "epoch={}-step={}".format(
                        item["epoch"],
                        item["step"],
                    ),
                    "{}.png".format(item["image_name"].replace("/", "_"))
                )
                os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                torchvision.utils.save_image(
                    image,
                    image_output_path,
                )
            except:
                traceback.print_exc()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, name="test")

    def configure_optimizers(self):
        # gaussian_model.training_setup() must be called here, where parameters have been moved to GPUs
        self.gaussian_model.training_setup(self.hparams["gaussian"].optimization, self.trainer.datamodule.dataparser_outputs.camera_extent)

        # initialize lists that store optimizers and schedulers
        optimizers = [
            self.gaussian_model.optimizer,
        ]
        schedulers = []

        def add_optimizers_and_schedulers(new_optimizers, new_schedulers):
            nonlocal optimizers
            nonlocal schedulers

            if new_optimizers is not None:
                if isinstance(new_optimizers, list):
                    optimizers += new_optimizers
                else:
                    optimizers.append(new_optimizers)
            if new_schedulers is not None:
                if isinstance(new_schedulers, list):
                    schedulers += new_schedulers
                else:
                    schedulers.append(new_schedulers)

        # renderer optimizer and scheduler setup
        renderer_optimizer, renderer_scheduler = self.renderer.training_setup(self)
        add_optimizers_and_schedulers(renderer_optimizer, renderer_scheduler)

        return optimizers, schedulers

    def save_gaussians(self):
        is_mp_strategy = isinstance(self.trainer.strategy, internal.mp_strategy.MPStrategy)
        if self.trainer.global_rank != 0 and is_mp_strategy is False:
            return

        if self.hparams["save_ply"] is True:
            # save ply file
            filename = "point_cloud.ply"
            # if self.trainer.global_rank != 0:
            #     filename = "point_cloud_{}.ply".format(self.trainer.global_rank)
            with torch.no_grad():
                output_dir = os.path.join(self.hparams["output_path"], "point_cloud",
                                          "iteration_{}".format(self.trainer.global_step))
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)
                self.gaussian_model.save_ply(output_path + ".tmp")
                os.rename(output_path + ".tmp", output_path)

            print("Gaussians saved to {}".format(output_path))

        # save checkpoint
        checkpoint_name_suffix = ""
        if is_mp_strategy is True:
            checkpoint_name_suffix = f"-rank={self.global_rank}"

        checkpoint_path = os.path.join(
            self.hparams["output_path"],
            "checkpoints",
            "epoch={}-step={}{}.ckpt".format(self.trainer.current_epoch, self.trainer.global_step, checkpoint_name_suffix),
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self.trainer.save_checkpoint(checkpoint_path)
        with torch.no_grad():
            xyz = self.gaussian_model.get_xyz
            rgb = eval_sh(0, self.gaussian_model.get_features[:, :1, :].transpose(1, 2), None)
            store_ply(os.path.join(
                self.hparams["output_path"],
                "checkpoints",
                "epoch={}-step={}{}-xyz_rgb.ply".format(self.trainer.current_epoch, self.trainer.global_step, checkpoint_name_suffix),
            ), xyz.cpu().numpy(), ((rgb + 0.5).clamp(min=0., max=1.) * 255).to(torch.int).cpu().numpy())
        print("Checkpoint saved to {}".format(checkpoint_path))

    def set_datamodule_device(self, device):
        if self.trainer is None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        datamodule.set_device(device)

    def _on_device_updated(self):
        self.metric.on_parameter_move(device=self.device)
        self.set_datamodule_device(self.device)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)

        self._on_device_updated()

        return self

    def cuda(self, device: Optional[Union[torch.device, int]] = None) -> Self:
        super().cuda(device)

        self._on_device_updated()

        return self

    def cpu(self) -> Self:
        super().cpu()

        self._on_device_updated()

        return self
