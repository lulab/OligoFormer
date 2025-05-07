# OligoFormer

[![python >3.8.20](https://img.shields.io/badge/python-3.8.20-brightgreen)](https://www.python.org/) 

Gene silencing through RNA interference (RNAi) has emerged as a powerful tool for studying gene function and developing therapeutics[[1]](#ref1). Small interfering RNA (siRNA) molecules play a crucial role in RNAi by targeting specific mRNA sequences for degradation. Identifying highly efficient siRNA molecules is essential for successful gene silencing experiments and therapeutic applications. Built on the transformer architecture[[2]](#ref2),  OligoFormer can capture multi-dimensional features and learn complex patterns of siRNA-mRNA interactions for siRNA efficacy prediction.

## Datasets

OligoFormer was trained on a dataset of mRNA and siRNA pairs with experimentally measured efficacy by Huesken et al[[3]](#ref3). The training data consisted of diverse mRNA sequences and corresponding siRNA molecules with known efficacies.

| dataset                                                      | siRNA number | cell  line              | 
| ------------------------------------------------------------ | ------------ | ----------------------- | 
| [Huesken](https://www.nature.com/articles/nbt1118)           | 2431         | H1299                   |
| [Reynolds](https://www.nature.com/articles/nbt936)           | 240          | HEK293                  | 
| [Vickers](https://www.jbc.org/article/S0021-9258(19)32641-9/fulltext) | 76           | T24                     | 
| [Haborth](https://www.liebertpub.com/doi/10.1089/108729003321629638) | 44           | HeLa                    |     
| [Ui-Tei](https://academic.oup.com/nar/article/32/3/936/2904484?login=false) | 62           |             HeLa                           |
| [Khvorova](https://www.nature.com/articles/nbt936)           | 14           | HEK293                  | 
| [Hiesh](https://academic.oup.com/nar/article/32/3/893/2904476) | 108          | HEK293T                 | 
| [Amarzguioui](https://pubmed.ncbi.nlm.nih.gov/12527766/)     | 46           | Cos-1,  HaCaT           |
| [Takayuki](https://academic.oup.com/nar/article/35/4/e27/1079934) | 702          | HeLa                    | 

## Model

![OligoFormer_architecture](figures/Figure1.png)

## Installation

**Implementation manual**

[English version](https://docs.qq.com/doc/DQm9GeUdSS0FBcUFY)

[Chinese version](https://docs.qq.com/doc/DQlJneHVvRkJIbE1Z)


### OligoFormer environment

Download the repository and create the environment of RNA-FM.

```bash
#Clone the OligoFormer repository from GitHub
git clone https://github.com/lulab/OligoFormer.git
cd ./OligoFormer
#Install the required dependencies
conda create -n oligoformer python=3.8*
```

### RNA-FM environment

source 1:  Download the packaged RNA-FM.

```bash
wget https://cloud.tsinghua.edu.cn/f/46d71884ee8848b3a958/?dl=1 -O RNA-FM.tar.gz
tar -zxvf RNA-FM.tar.gz
```

source 2:  Create the environment of RNA-FM[[4]](#ref4).

```bash
git clone https://github.com/ml4bio/RNA-FM.git
cd ./RNA-FM
conda env create --name RNA-FM -f environment.yml
```
Download pre-trained models from [this gdrive link](https://drive.google.com/drive/folders/1VGye74GnNXbUMKx6QYYectZrY7G2pQ_J?usp=share_link) and place the pth files into the `pretrained` folder.


## Usage

You should have at least an NVIDIA GPU and a driver on your system to run the training or inference.

### 1.Activate the created conda environment

```bash
source activate oligoformer
pip install -r requirements.txt
```

### 2.Model training

```bash
#The following command take ~60 min on a V100 GPU
python scripts/main.py --datasets Hu Mix --cuda 0 --learning_rate 0.0001 --batch_size 16 --epoch 200 --early_stopping 30
```

### 3.Model inference

#### 3.1 Inference without off-target


Option 1:  Input the fasta file of mRNA sequence (Traverse mRNA with 19nt window size).

```bash
python scripts/main.py --infer 1 -i1 data/example.fa
```

Option 2:  Input the fasta files of the mRNA and specific siRNAs (only predict these specific siRNAs).

```bash
python scripts/main.py --infer 1 -i1 data/example.fa -i2 data/example_siRNA.fa
```

Option 3:  Input the mRNA sequence manually.

```bash
python scripts/main.py --infer 2
```


#### 3.2 Inference with off-target

<img src="figures/Figure5.jpg" alt="Off-target pipeline" width="400"/>


- Dependency of perl

source 1: CPAN

```bash
cpan Statistics::Lite
cpan Bio::TreeIO
# You also need install Vienarna package and export the PATH, and adjust the perl5lib to your own path.
# You need provide the ORF and UTR fatsa of mRNA to predict the off-target effects. The order of the sequence needs to be consistent across both files. Refer to the example data.
```

source 2: Download

```bash
wget https://cloud.tsinghua.edu.cn/f/cab2afdf951140a48fec/?dl=1 -O PerlLib.zip
unzip PerlLib.zip
export PERL5LIB=$(pwd)/PerlLib:$PERL5LIB
```

- Replace path
```bash
cd off-target/pita && make install && cd ../../
```

- Command

```bash
python scripts/main.py --infer 1 -i1 ./data/example.fa -off -tox -a
python scripts/main.py --infer 1 -i1 ./data/example.fa -off -tox -a -top 100 # only calculate off-target effects of top 100(customed) effective siRNAs
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
    ```bash
    docker pull yilanbai/oligoformer:v1.0
    ```
    source 2: Aliyun
    ```bash
    docker pull registry.cn-hangzhou.aliyuncs.com/yilanbai/oligoformer:v1.0
    ```
    source 3: Tsinghua Cloud
   
    [Download Link](https://cloud.tsinghua.edu.cn/f/2cca306e868a4b7897d3/)

3. **Run the Docker Container**:

    ```bash
    docker run -it --name oligoformer-container -dt --restart unless-stopped yilanbai/oligoformer:v1.0 && docker exec -it oligoformer-container bash
    ```

4. **Access the OligoFormer Tool**:

    Once inside the container, you can start using OligoFormer with the following command:

    ```bash
    oligoformer -h # help
    oligoformer # infer
    oligoformer -i 1 -i1 data/example.fa -i2 data/example_siRNA.fa # infer only interested siRNA(faster)
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

[1]<a name="ref1"></a> [Zamore, Phillip D., et al. "RNAi: double-stranded RNA directs the ATP-dependent cleavage of mRNA at 21 to 23 nucleotide intervals." *cell* 101.1 (2000): 25-33.](https://www.sciencedirect.com/science/article/pii/S0092867400806200)

[2]<a name="ref2"></a> [Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017).](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

[3]<a name="ref3"></a> [Huesken, D., Lange, J., Mickanin, C. *et al.* Design of a genome-wide siRNA library using an artificial neural network. *Nat Biotechnol* **23**, 995–1001 (2005).](https://www.nature.com/articles/nbt1118#Abs1)

[4]<a name="ref4"></a> [Chen, Jiayang, et al. "Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions." arXiv preprint arXiv:2204.00300 (2022).](https://arxiv.org/abs/2204.00300)

## Citations
If you find the models useful in your research, we ask that you cite the relevant paper:

```bibtex
@article{bai2024oligoformer,
  title={OligoFormer: an accurate and robust prediction method for siRNA design},
  author={Bai, Yilan and Zhong, Haochen and Wang, Taiwei and Lu, Zhi John},
  journal={bioRxiv},
  pages={2024--02},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License and Disclaimer

Non-Commercial Use:

This tool may be used freely for academic research, non-profit projects, and personal use, provided all copyright notices are retained.

Commercial Use:

Any use for commercial purposes (including but not limited to commercial research, product development, or internal business operations) requires prior written authorization from Tsinghua University. Contact: ott@tsinghua.edu.cn.

For full terms, see the [LICENSE](./LICENSE) file.

