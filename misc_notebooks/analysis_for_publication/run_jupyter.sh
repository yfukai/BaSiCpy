#!/bin/bash
docker run -it --rm \
  -v "/mnt/showers2":"/mnt/showers2" \
  -v "/work":"/work" \
  -v "${PWD}":"${PWD}" \
  -w "${PWD}" \
  -v /usr/share/X11/xkb:/usr/share/X11/xkb \
  -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
  -v ${HOME}/.jupyter:${HOME}/.jupyter \
  -p8988:8988 \
  -e DISPLAY=$DISPLAY \
  -e TZ=Asia/Tokyo \
  --gpus=all \
  yfukai/basicpy-benchmark:latest 
 # -f $(id -gn) -g $(id -g) -t $(id -nu) -u $(id -u) -p ${HOME} \
 # -- jupyter lab --ip=0.0.0.0 --port 8988
