#!/usr/bin/env bash
python3 main.py --tag=../tags_extract --image=../illustrations_128 --cuda --batch=64 --epoch=500 --check_step=1 # --resume=./checkpoint/checkpoint_epoch_040.pth