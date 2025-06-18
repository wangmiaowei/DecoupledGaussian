data_path=../input_dataset/bear
res_path=../exp_res/bear

python utils/get_sam_masks.py $data_path/images
# # Rendering and Extract Mesh

python utils/get_sam_mask_scales.py --model_path=$res_path  --data_path=$data_path
python seganygs.py fit \
    --config configs/segany_splatting.yaml \
    --data.path  $data_path  \
    --model.initialize_from $res_path \
    -n $res_path -v seganygs \
    --viewer
python viewer.py $res_path/seganygs

#python viewer.py /home/wmw/project/codespace/PhysGaussian/model/vasedeck_whitebg-trained/point_cloud/iteration_30000/point_cloud.ply
#python viewer.py /mnt/wmw/exp_res/PGSR/bear --vanilla_seganygs

#/mnt/wmw/exp_res/PGSR/figurines/cameras.json

#python seganygs.py fit --config configs/segany_splatting.yaml --data.path /mnt/wmw/dataset/figurines --model.initialize_from /home/wmw/project/exp_res/PGSR/figurines  -n /home/wmw/project/exp_res/PGSR/figurines -v seganygs  