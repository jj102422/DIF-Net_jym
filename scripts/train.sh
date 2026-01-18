gpu=0
name=dif-net_CTSpine1K（6）
dst_list=knee_cbct
n_view=2

mkdir -p ./logs/$name

CUDA_VISIBLE_DEVICES=$gpu python code/train.py \
    --name $name \
    --batch_size 4 \
    --epoch 400 \
    --dst_list $dst_list \
    --num_views $n_view \
    --combine mlp
