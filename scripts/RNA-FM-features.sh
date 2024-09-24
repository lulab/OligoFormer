#!/bin/bash
cd ./RNA-FM/redevelop
for file in Hu_siRNA Hu_mRNA Mix_siRNA Mix_mRNA Taka_siRNA Taka_mRNA 
do
python launch/predict.py --config="pretrained/extract_embedding.yml" \
--data_path=../../data/fasta/$file.fa --save_dir=../../data/RNAFM/$file \
--save_frequency 1 --save_embeddings
done
cd ../../
