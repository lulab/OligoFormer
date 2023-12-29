# OligoFormer

## Overview

Gene silencing through RNA interference (RNAi) has emerged as a powerful tool for studying gene function and developing therapeutics[1]. Small interfering RNA (siRNA) molecules play a crucial role in RNAi by targeting specific mRNA sequences for degradation. Identifying highly efficient siRNA molecules is essential for successful gene silencing experiments and therapeutic applications.

OligoFormer is built on the transformer architecture[2], a state-of-the-art deep learning model that has revolutionized various natural language processing tasks. By adapting the transformer model for siRNA design, OligoFormer can effectively capture sequence features and learn complex patterns associated with siRNA efficacy.

## Features

- **siRNA Design**: OligoFormer uses the input mRNA sequence and generates candidate siRNA molecules with complementary sequences.
- **siRNA Efficacy Prediction**: OligoFormer predicts the efficacy of each candidate siRNA molecule for targeting the given mRNA sequence.
- **RBP & ic-Shape incorporation**: OligoFormer combines information of RBP and ic-Shape from POSTAR3[3].
- **High Performance**: OligoFormer achieves state-of-the-art performance in siRNA efficacy prediction, providing accurate and reliable results.

## Datasets

OligoFormer was trained on a dataset of mRNA and siRNA pairs with experimentally measured efficacy by Huesken et al[4]. The model was trained using supervised learning, where the objective was to minimize the discrepancy between predicted efficacy and the actual experimental efficacy.

The training data consisted of diverse mRNA sequences and corresponding siRNA molecules with known efficacies. Care was taken to ensure the dataset's quality and to remove any biases that might affect the model's generalization ability.

| dataset                                                      | siRNA number | cell  line              | additional  info                          |
| ------------------------------------------------------------ | ------------ | ----------------------- | ----------------------------------------- |
| [Huesken](https://www.nature.com/articles/nbt1118)           | 2431         | H1299                   | Human  NCI-H1299 cells obtained from ATCC |
| [Shabalina](https://europepmc.org/article/PMC/1431570)       | 653          | HEK293,  T24, HeLa      | Summary  data                             |
| [Reynolds](https://www.nature.com/articles/nbt936)           | 240          | HEK293                  | HEK293  or HEK293-Luc cells               |
| [Vickers](https://www.jbc.org/article/S0021-9258(19)32641-9/fulltext) | 76           | T24                     | ATCC,  Manassas,  VA                      |
| [Haborth](https://www.liebertpub.com/doi/10.1089/108729003321629638) | 44           | HeLa                    |                                           |
| [Takayuki](https://academic.oup.com/nar/article/35/4/e27/1079934) | 702          | HeLa                    | HeLa  cells stably expressing Hyg/EGFP    |
| [Ui-](https://academic.oup.com/nar/article/32/3/936/2904484?login=false)[Tei](https://academic.oup.com/nar/article/32/3/936/2904484?login=false) | 62           | CHO-K1,  HeLa,  E14TG2a |                                           |
| [Khvorova](https://www.nature.com/articles/nbt936)           | 14           | HEK293                  |                                           |
| [Hiesh](https://academic.oup.com/nar/article/32/3/893/2904476) | 108          | HEK293T                 |                                           |
| [Amarzguioui](https://pubmed.ncbi.nlm.nih.gov/12527766/)     | 46           | Cos-1,  HaCaT           |                                           |

## Model

![OligoFormer_architecture](figures/Figure1.pdf)

## Training

OligoFormer can be used through a programming interface or a command-line interface (CLI). Follow the instructions below to set up and utilize the model.

### Prerequisites

- Python 3.7
- Tensorflow 2.0 or higher
- Transformers library

### Installation

```bash
git clone https://github.com/byl18/OligoFormer.git #Clone the OligoFormer repository from GitHub
pip install -r requirements.txt #Install the required dependencies
```

### Usage

```python
# unrealized
import OligoFormer #Import the OligoFormer module
model = OligoFormer.Model() #Initialize the OligoFormer model
mRNA_sequence = "AUGCUACGAUUGCGACUUUGU"
candidate_siRNAs = model.design_siRNA(mRNA_sequence) #Design siRNA molecules
efficacies = model.predict_efficacy(candidate_siRNAs) #Predict siRNA efficacy
best_siRNA = max(candidate_siRNAs, key=efficacies.get) #Retrieve the siRNA with the highest predicted efficacy
```

OligoFormer also provides a CLI for easy access to siRNA design and efficacy prediction. Here's an example of how to use it:

```bash
python script/main.py --datasets ['Hu','Sha'] --output_dir output_dir --new_model True --batch_size 512 --epoch 300 --weight_decay 0.9999 --early_stopping 10 #default params
```



## Results

<img src="src/merge_AUROC.png" width = "360" height = "360" alt="merge_AUROC" align=center /><img src="figures/Figure2.png" width = "360" height = "360" alt="merge_AUPRC" align=center />

<img src="src/merge_AUROC.png" width = "360" height = "360" alt="merge_AUROC" align=center /><img src="figures/Figure3.png" width = "360" height = "360" alt="merge_AUPRC" align=center />

## Limitations

- The size of training dataset is limited and more siRNA datasets may improve this model further.
- Different datasets have strong batch effects due to different experimental conditions. So it may not perform optimally on mRNA sequences that significantly differ from the training data.

## Conclusion

OligoFormer is a transformer-based model designed to identify highly efficient siRNA molecules for gene silencing experiments. By leveraging the power of deep learning and natural language processing techniques, OligoFormer provides accurate siRNA design and efficacy prediction capabilities. It is a valuable tool for researchers and scientists working in the field of gene expression modulation and therapeutics. For more details, refer to the [OligoFormer GitHub repository](https://github.com/byl18/OligoFormer).

## References

[1] [Zamore, Phillip D., et al. "RNAi: double-stranded RNA directs the ATP-dependent cleavage of mRNA at 21 to 23 nucleotide intervals." *cell* 101.1 (2000): 25-33.](https://www.sciencedirect.com/science/article/pii/S0092867400806200)

[2] [Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017).](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

[3] [Zhao, Weihao, et al. "POSTAR3: an updated platform for exploring post-transcriptional regulation coordinated by RNA-binding proteins." *Nucleic Acids Research* 50.D1 (2022): D287-D294.](https://academic.oup.com/nar/article/50/D1/D287/6353804)

[4] [Huesken, D., Lange, J., Mickanin, C. *et al.* Design of a genome-wide siRNA library using an artificial neural network. *Nat Biotechnol* **23**, 995–1001 (2005).](https://www.nature.com/articles/nbt1118#Abs1)

