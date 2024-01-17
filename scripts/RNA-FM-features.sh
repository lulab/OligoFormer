#!/bin/bash
path=$(which python)

path=(${path//\/envs/ }) 
cd ./RNA-FM/redevelop
for file in Hu_siRNA Hu_mRNA new_siRNA new_mRNA Taka_siRNA Taka_mRNA 
do
$path/envs/RNA-FM/bin/python launch/predict.py --config="pretrained/extract_embedding.yml" \
--data_path=../../data/fasta/$file.fa --save_dir=../../data/RNAFM/$file \
--save_frequency 1 --save_embeddings
done
cd ../../
