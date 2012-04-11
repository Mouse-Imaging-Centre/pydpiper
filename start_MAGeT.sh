#!/bin/bash
PYD_HOME=/home/jp/git/pydpiper
PATH=$PATH:$PYD_HOME/pydpiper:$PYD_HOME/applications/MAGeT_human
PYTHONPATH=$PYTHONPATH:$PYD_HOME
ARGS="--atlas-labels models/mask_right_oncolinnl_7.mnc \
      --atlas-labels models/mask_left_oncolinnl_7.mnc \
      --atlas-image models/colin27_t1_tal_lin.mnc  \
      --atlas-roi models/colin_bg_generous_0.3mm.mnc \
      --create-graph \
      --output-dir output \
      --max-templates 2 \
      --queue=sge \
      --num-executors 0 \

      /home/jp/git/pydpiper/input/"
python $PYD_HOME/applications/MAGeT_human/MAGeT.py $ARGS
