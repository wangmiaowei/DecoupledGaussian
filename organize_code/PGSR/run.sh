# Training
python train.py -s /mnt/wmw/dataset/garden -m /mnt/wmw/exp_res/PGSR/garden --max_abs_split_points 0 --opacity_cull_threshold 0.05

# Rendering and Extract Mesh

#python render.py -m /mnt/wmw/exp_res/PGSR/truck --max_depth 10.0 --voxel_size 0.01