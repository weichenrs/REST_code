CONFIG=$1
GPUS=$2
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-1}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"10.176.242.63"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_ds.py \
    $CONFIG \
    --launcher pytorch ${@:3}
