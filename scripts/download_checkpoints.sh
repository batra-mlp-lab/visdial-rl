#!/usr/bin/env bash

# SL checkpoints
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_sl_ep60.vd
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_sl_ep60.vd

# SL-Delta checkpoints
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_sl_ep15_delta.vd
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_sl_ep15_delta.vd


# RL checkpoints
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep10.vd
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep20.vd

wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep10.vd
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep20.vd

# RL-Delta checkpoints
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep10_delta.vd
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep20_delta.vd

wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep10_delta.vd
wget -P checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep20_delta.vd
