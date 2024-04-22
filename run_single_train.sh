#!/bin/bash
if [ -z "$1" ]
then
    echo "Missing network (network)"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Missing optimizer (opt)"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Missing dataset (dts)"
    exit 1
fi

docker run -e argg="1" -v /home/lorenzo/Projects/CMA_light_code:/work/project/ -v /home/lorenzo/Results_CMA:/work/results/ -v /home/lorenzo/Datasets/CMA_L:/work/datasets/ --gpus all --ipc host -u 1001:1001 lorenzopapa5/cuda12.1.0-python3.8-pytorch2.1.0 /usr/bin/python3 /work/project/main_Corrado.py --network "${1}" --opt "${2}" --dts "${3}"
