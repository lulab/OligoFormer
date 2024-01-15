#!/bin/bash
path=$(which python)

path=(${path//\/envs/ }) 
cd ./RNA-FM/redevelop
$path/envs/RNA-FM/bin/python launch/predict.py --config="pretrained/extract_embedding.yml" \
--data_path=$1/mRNA.fa --save_dir=$1/mRNA \
--save_frequency 1 --save_embeddings

$path/envs/RNA-FM/bin/python launch/predict.py --config="pretrained/extract_embedding.yml" \
--data_path=$1/siRNA.fa --save_dir=$1/siRNA \
--save_frequency 1 --save_embeddings

cd ../../
