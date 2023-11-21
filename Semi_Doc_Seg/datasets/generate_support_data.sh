DATA_ROOT=/data2/users/abanerjee/HJDataset

cd coco

ln -s $DATA_ROOT/train ./
ln -s $DATA_ROOT/val ./
ln -s $DATA_ROOT/annotations ./

python3 1_split_filter.py ./ 
#python3 2_balance.py ./
python3 3_gen_support_pool.py ./
python3 4_gen_support_pool_10_shot.py ./

