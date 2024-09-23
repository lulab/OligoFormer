# OligoFormer

[![python >3.7.16](https://img.shields.io/badge/python-3.7.16-brightgreen)](https://www.python.org/) 

Gene silencing through RNA interference (RNAi) has emerged as a powerful tool for studying gene function and developing therapeutics[1]. Small interfering RNA (siRNA) molecules play a crucial role in RNAi by targeting specific mRNA sequences for degradation. Identifying highly efficient siRNA molecules is essential for successful gene silencing experiments and therapeutic applications. Built on the transformer architecture[2],  OligoFormer can capture multi-dimensional features and learn complex patterns of siRNA-mRNA interactions for siRNA efficacy prediction.

## Datasets

OligoFormer was trained on a dataset of mRNA and siRNA pairs with experimentally measured efficacy by Huesken et al[4]. The training data consisted of diverse mRNA sequences and corresponding siRNA molecules with known efficacies.

| dataset                                                      | siRNA number | cell  line              | 
| ------------------------------------------------------------ | ------------ | ----------------------- | 
| [Huesken](https://www.nature.com/articles/nbt1118)           | 2431         | H1299                   |
| [Reynolds](https://www.nature.com/articles/nbt936)           | 240          | HEK293                  | 
| [Vickers](https://www.jbc.org/article/S0021-9258(19)32641-9/fulltext) | 76           | T24                     | 
| [Haborth](https://www.liebertpub.com/doi/10.1089/108729003321629638) | 44           | HeLa                    |     
| [Ui-](https://academic.oup.com/nar/article/32/3/936/2904484?login=false)[Tei](https://academic.oup.com/nar/article/32/3/936/2904484?login=false) | 62           |             HeLa                           |
| [Khvorova](https://www.nature.com/articles/nbt936)           | 14           | HEK293                  | 
| [Hiesh](https://academic.oup.com/nar/article/32/3/893/2904476) | 108          | HEK293T                 | 
| [Amarzguioui](https://pubmed.ncbi.nlm.nih.gov/12527766/)     | 46           | Cos-1,  HaCaT           |
| [Takayuki](https://academic.oup.com/nar/article/35/4/e27/1079934) | 702          | HeLa                    | 

## Model

![OligoFormer_architecture](figures/Figure1.png)

## Installation

### OligoFormer environment

Download the repository and create the environment of RNA-FM.

```bash
#Clone the OligoFormer repository from GitHub
git clone https://github.com/lulab/OligoFormer.git
cd ./OligoFormer
#Install the required dependencies
conda env create -n oligoformer -f environment.yml
```

### RNA-FM environment

Download the repository and create the environment of RNA-FM.
```
git clone https://github.com/ml4bio/RNA-FM.git
cd ./RNA-FM
conda env create --name RNA-FM -f environment.yml
```

Download pre-trained models from [this gdrive link](https://drive.google.com/drive/folders/1VGye74GnNXbUMKx6QYYectZrY7G2pQ_J?usp=share_link) and place the pth files into the `pretrained` folder.



## Usage

You should have at least an NVIDIA GPU and a driver on your system to run the training or inference.

### 1.Activate the created conda environment

```source activate oligoformer```

### 2.Model training

```
#The following command take ~30 min on a V100 GPU
python scripts/main.py --datasets Hu Mix --cuda 0 --learning_rate 0.0001 --batch_size 16 --epoch 100 --early_stopping 30
```

### 3.Model inference

#### 3.1 Inference without off-target

```
python scripts/main.py --infer 1 --infer_fasta ./data/example.fa --infer_output ./result/
```

- Example output

```text
pos sense siRNA efficacy
0 UGAAUUUUUGUCAGAUAAA UUUAUCUGACAAAAAUUCA 0.9139741711020469
1 GAAUUUUUGUCAGAUAAAU AUUUAUCUGACAAAAAUUC 0.8864658409953117
2 AAUUUUUGUCAGAUAAAUA UAUUUAUCUGACAAAAAUU 0.815981000483036
3 AUUUUUGUCAGAUAAAUAA UUAUUUAUCUGACAAAAAU 0.8179122650027275
4 UUUUUGUCAGAUAAAUAAA UUUAUUUAUCUGACAAAAA 0.7880132337212562
5 UUUUGUCAGAUAAAUAAAA UUUUAUUUAUCUGACAAAA 0.7990648913383483
6 UUUGUCAGAUAAAUAAAAU AUUUUAUUUAUCUGACAAA 0.7055106237530708
7 UUGUCAGAUAAAUAAAAUA UAUUUUAUUUAUCUGACAA 0.7850472775697708
8 UGUCAGAUAAAUAAAAUAA UUAUUUUAUUUAUCUGACA 0.8157202693819999
9 GUCAGAUAAAUAAAAUAAA UUUAUUUUAUUUAUCUGAC 0.8842068641781807

# pos: start position of siRNA at mRNA
# sense: sense strand sequence, complimentary to siRNA
# siRNA: siRNA sequence
# efficacy: The predicted efficacy of siRNA
```

#### 3.2 Inference with off-target

- Dependency of perl

```
cpan Statistics::Lite
cpan Bio::TreeIO
# You also need install Vienarna package and export the PATH, and adjust the perl5lib to your own path.
# You need provide the ORF and UTR fatsa of mRNA to predict the off-target effects. The order of the sequence needs to be consistent across both files. Refer to the example data.
```

- Command

```
python scripts/main.py --infer 1 --infer_fasta ./data/example.fa --infer_output ./result/ -off -tox
```

- Example output

```text
pos,sense,siRNA,efficacy,PITA_Score,Scaled_PITA_Score,PheLiM_Score,Scaled_PheLiM_Score,off_target_score,Immune_Motif,evaluation_score
0,UGAAUUUUUGUCAGAUAAA,UUUAUCUGACAAAAAUUCA,0.9139741711020468,-10.07,1.0,0.28300000000000003,0.2866379310344829,0.671262782443687,0,0.5351767612351033
1,GAAUUUUUGUCAGAUAAAU,AUUUAUCUGACAAAAAUUC,0.8864658409953117,-8.17,0.6207584830339321,0.15,0.0,0.0,0,1.0
2,AAUUUUUGUCAGAUAAAUA,UAUUUAUCUGACAAAAAUU,0.8159810004830359,-8.96,0.7784431137724552,0.23399999999999999,0.1810344827586207,0.4425563810483377,0,0.4636817856550889
3,AUUUUUGUCAGAUAAAUAA,UUAUUUAUCUGACAAAAAU,0.8179122650027275,-7.7,0.5269461077844312,0.5660000000000001,0.8965517241379312,1.0,0,0.20202874870817145
4,UUUUUGUCAGAUAAAUAAA,UUUAUUUAUCUGACAAAAA,0.7880132337212562,-7.4,0.4670658682634731,0.368,0.4698275862068966,0.705735069085299,0,0.26680191635745165
5,UUUUGUCAGAUAAAUAAAA,UUUUAUUUAUCUGACAAAA,0.7990648913383483,-7.58,0.502994011976048,0.38,0.49568965517241387,0.7522461644008175,0,0.26754605953645344
6,UUUGUCAGAUAAAUAAAAU,AUUUUAUUUAUCUGACAAA,0.7055106237530708,-7.63,0.5129740518962076,0.5720000000000001,0.9094827586206898,0.9882482133646434,0,0.0
7,UUGUCAGAUAAAUAAAAUA,UAUUUUAUUUAUCUGACAA,0.7850472775697708,-7.37,0.46107784431137727,0.614,1.0,0.9508590794451448,0,0.15983727224740343
8,UGUCAGAUAAAUAAAAUAA,UUAUUUUAUUUAUCUGACA,0.8157202693819999,-6.53,0.2934131736526947,0.61,0.9913793103448276,0.682184317952127,0,0.33302818248637234
9,GUCAGAUAAAUAAAAUAAA,UUUAUUUUAUUUAUCUGAC,0.8842068641781807,-5.06,0.0,0.19899999999999998,0.10560344827586204,0.0,0,0.9941495828005047

# pos: start position of siRNA at mRNA
# sense: sense strand sequence, complimentary to siRNA
# siRNA: siRNA sequence
# efficacy: The predicted efficacy of siRNA
# PITA_Score: The PITA score of siRNA
# Scaled_PITA_Score: The scaled PITA score of siRNA (0-1)
# PheLiM_Score: The PheLiM score of siRNA
# Scaled_PheLiM_Score: The scaled PheLiM score of siRNA (0-1)
# off_target_score: The off-target score of siRNA, which is the harmonic average of Scaled_PITA_Score and Scaled_PheLiM_Score
# Immune_Motif: The immune induced motif number of siRNA
# evaluation_score: The evaluation score of siRNA, which is calculated by efficacy and off_target_score
```

## User-friendly Docker image

![Docker](https://img.shields.io/badge/Docker-Ready-blue)

The Docker image simplifies the installation and setup process, making it easy for users to get started with OligoFormer without worrying about dependencies and environment configuration.

## Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine.

## Installation

1. **Pull the Docker Image**:

    
    You just need to choose one source.

    source 1: DockerHub
    ```sh
    docker pull yilanbai/oligoformer:v1.0
    ```
    source 2: Aliyun
    ```sh
    docker pull registry.cn-hangzhou.aliyuncs.com/yilanbai/oligoformer:v1.0
    ```
    source 3: Tsinghua Cloud
   
    [Download Link](https://cloud.tsinghua.edu.cn/f/2cca306e868a4b7897d3/)

3. **Run the Docker Container**:

    ```sh
    docker run -it --name oligoformer-container -dt --restart unless-stopped yilanbai/oligoformer:v1.0 && docker exec -it oligoformer-container bash
    ```

4. **Access the OligoFormer Tool**:

    Once inside the container, you can start using OligoFormer with the following command:

    ```sh
    oligoformer -h # help
    oligoformer # infer
    oligoformer -i 1 -i1 example.fa -i2 example_siRNA.fa # infer only interested siRNA(faster)
    oligoformer -off # infer with off-target prediction
    oligoformer -tox # infer with toxicity prediction
    oligoformer -off -tox # infer with off-target and toxicity prediction
    oligoformer -m 2 # mismatch input 19nt siRNA
    oligoformer -i 0 -t # test inter-dataset
    oligoformer -i 0 -s -t # test intra-dataset
    # We recommand you to run the following two commands on the patform with GPUs.
    oligoformer -i 0 # train inter-dataset
    oligoformer -i 0 -s # train intra-dataset
    
## References

[1] [Zamore, Phillip D., et al. "RNAi: double-stranded RNA directs the ATP-dependent cleavage of mRNA at 21 to 23 nucleotide intervals." *cell* 101.1 (2000): 25-33.](https://www.sciencedirect.com/science/article/pii/S0092867400806200)

[2] [Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017).](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

[3] [Zhao, Weihao, et al. "POSTAR3: an updated platform for exploring post-transcriptional regulation coordinated by RNA-binding proteins." *Nucleic Acids Research* 50.D1 (2022): D287-D294.](https://academic.oup.com/nar/article/50/D1/D287/6353804)

[4] [Huesken, D., Lange, J., Mickanin, C. *et al.* Design of a genome-wide siRNA library using an artificial neural network. *Nat Biotechnol* **23**, 995–1001 (2005).](https://www.nature.com/articles/nbt1118#Abs1)

## License and Disclaimer
This tool is for research purpose and not approved for clinical use. The tool shall not be used for commercial purposes without permission.

