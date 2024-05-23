#!/bin/sh

export CUDA_VISIBLE_DEVICES=$(for d in $(echo "$CUDA_VISIBLE_DEVICES" | sed 's/,/ /g') ; do nvidia-smi -L | grep -i $d | cut -d ':' -f 1 | cut -d ' ' -f 2 ; done | tr '\n' ',' | awk '{$1=substr($1, 1, length($1)-1);}1')

. /cluster/research-groups/deneke/minecraft-ml-pyenv/bin/activate
exec python "$@"