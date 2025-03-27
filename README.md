# [CVPR2025] DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction

### [[Project Page](https://wangmiaowei.github.io/DecoupledGaussian.github.io/)] [[arXiv](https://arxiv.org/abs/2503.05484v1)] 

Miaowei Wang<sup>1</sup>\, Yibo Zhang<sup>2</sup>\, Rui Ma<sup>2</sup>\, Weiwei Xu<sup>3</sup>\, Changqing Zou<sup>3</sup>, Daniel Morris<sup>4</sup><br>
<sup>1</sup>The University of Edinburgh, <sup>2</sup>
Jilin University, <sup>3</sup>Zhejiang University, <sup>4</sup>Michigan State University <br>

![DecoupledGaussian](assets/teaser.png)
---

### Abstract

*We present DecoupledGaussian, a novel system that decouples static objects from their contacted surfaces captured in-the-wild videos, a key prerequisite for realistic Newtonian-based physical simulations. Unlike prior methods focused on synthetic data or elastic jittering along the contact surface, which prevent objects from fully detaching or moving independently, DecoupledGaussian allows for significant positional changes without being constrained by the initial contacted surface. Recognizing the limitations of current 2D inpainting tools for restoring 3D locations,
our approach uses joint Poisson fields to repair and expand the Gaussians of both objects and contacted scenes after separation. This is complemented by a multi-carve strategy to refine the object’s geometry. Our system enables realistic simulations of decoupling motions, collisions, and fractures driven by user-specified impulses, supporting complex interactions within and across multiple scenes. We validate DecoupledGaussian through a comprehensive user study and quantitative benchmarks. This system enhances digital interaction with objects and scenes in real-world environments, benefiting industries such as VR, robotics, and autonomous driving.*

---
## 📑 Open-Source Plan

- [x] Teaser Simulation Cases
- [ ] Upload Other Simulation Cases
- [ ] Gaussian Preparation Code
- [ ] Gaussian Restoration Code
      
## Environment Setup
```bash
# Clone Repository 
git clone https://github.com/wangmiaowei/DecoupledGaussian.git

# Navigate to the project directory
cd DecoupledGaussian

# Install required Python packages
pip install -r requirements.txt
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```

---

## Interactive Simulation

Our algorithm has been tested on multiple datasets. Below is an example simulation for the *bear* object as illustrated in **Figure 1**.

```bash
# Navigate to the dataset folder
cd dataset/bear_data

# Download GS models using the anonymous links
gdown https://drive.google.com/file/d/1HoOgTgajUsrLujIk6PJmZ084J_GKmEj_/view?usp=drive_link
gdown https://drive.google.com/file/d/1m9DUDEINvOza9qxgPqvyjvEBcSrbkDVx/view?usp=drive_link

# Return to the main directory
cd ../..

# Run the Teaser Simulation
python gs_simulation.py \
--object_path dataset/bear_data/bear_obj_0.ply \
--scene_path dataset/bear_data/bear_scene_0.ply \
--output_path result_bear \
--config config/bear_config.json \
--render_img --compile_video

# Access the rendered video "output.mp4"
cd result_bear 

```
![Demo](result_bear/bear_collision_compressed.gif)


---
## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @InProceedings{decoupledGaussian,
    title={DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction},
    author={Wang, Miaowei and Zhang, Yibo and Ma, Rui and Xu, Weiwei and Zou, Changqing and Morris, Daniel},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}}
   ```


## Acknowledgement

This simulation code is built upon [PhysGaussian](https://github.com/XPandora/PhysGaussian). Thanks to the contributors for their great work.





