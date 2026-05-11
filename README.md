# 🧬 TSMGen: Target-Specific Molecule Generation via Higher-Order Structural Dependencies and Context-Aware Bidirectional Fusion 

[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Official%20Implementation-orange.svg)](https://pytorch.org/)

The code is an official PyTorch-based implementation in the paper [TSMGen: Target-Specific Molecule Generation via Higher-Order Structural Dependencies and Context-Aware Bidirectional Fusion  ](https://icml.cc/virtual/2026/poster/63863) (Accepted in International Conference on Machine Learning , 2026).

## 📖 Abstract


Efficiently designing high-quality molecules targeting disease-relevant targets is a critical challenge. Most existing methods can capture pairwise amino acid relations, neglecting the higher-order relations among multiple amino acids. This paper proposes a target-specific molecule generation framework, namely TSMGen, to comprehensively capture the local and global structural information of the protein pocket by modeling higher-order spatial dependencies both at the atomic and the amino acid levels. Furthermore, we design a context-aware bidirectional fusion module to learn the more detailed structural information about the protein pocket. This module simultaneously attends to features from both the protein pocket and the molecule, fully leveraging the structural information from both to optimize the generation process of targeted molecules, thereby enhancing the quality of generated molecules. Experiments show that TSMGen outperforms state-of-the-art methods in terms of Vina Score, High Affinity, QED, SA and Diversity, and a case study on $\beta$-secretase enzyme further confirms its ability to generate molecules with stronger binding affinity.
![model](./assert/model_202605060001.png)


## ⚙️ Installation

### 1. Hardware & Environment Requirements

The code has been developed and tested on the following hardware configuration:

| Component | Specification |
| :--- | :--- |
| **Operating System** | Ubuntu 20.04 |
| **GPU** | NVIDIA GeForce RTX 3090 (24GB VRAM) |
| **CUDA Version** | 11.3 |
| **Memory (RAM)** | 64 GB |

---

### 2. Create a new conda environment
```bash
conda env create -f requirements.yml
conda activate tsmgen
```

## 📂 Dataset

The model is trained on the CrossDocked dataset.

### 1. Download the dataset:

Download  [here](https://drive.google.com/file/d/1BTbuD45VBkBoPAVdNNthdzq1f2D1sAjP/view?usp=sharing).

### 2. Extract it to your preferred location:

Unzip the file to a suitable location on your machine:
```bash
unzip crossdocked_pocket10_mol2.zip -d /path/to/your/desired/location
```

### 3.Update the data path:

Update the data path in the training configuration file (`config/train.yaml`):
```yaml
# config/train.yaml
pocket_dir: /path/to/your/desired/location
```


## 🚀 Usage
### 1. Model Training
To start training the model, simply run:
```bash
python train_valid_valloss.py
```

### 2. Molecule Sampling

Generate molecules using the standard sampling script:
```bash
python sample.py
```


### 3. Docking & Evaluation Pipeline
We provide a complete pipeline in the ```evaluation_dock/``` folder to evaluate the generated molecules:
1. **Format Conversion:** 
Convert SMILES to PDBQT format using ```smiles2pdbqt.py``` 
2. **Molecular Docking:**
Run QVina docking using ```dock_qvina.py``` . Docking results will be saved in ```dock_file_save/``` under a timestamped folder.
3. **Metrics Analysis:**
   1. Calculate the high-affinity ratio: `python get_high_affintiy.py`
   2. Calculate the mean affinity value: `python get_vina_mean.py`



### 4. Case Study ($\beta$-secretase)

To reproduce the case study or run generation on specific proteins:
1. Download the target proteins from the [RCSB PDB website](https://www.rcsb.org/).
2. Use PyMOL to remove water molecules.
3. Convert the cleaned proteins to ```PDBQT``` format using OpenBabel.
4. Update parameters in ```config/sample_casestudy.yaml```.
5. Run the case study generation script:
```bash
python sample_casestudy.py
```


## 🗂️ Folder Structure  
```bash
TSMGen/
├── case_study/        # Case study configurations and specific docking files
├── config/            # YAML configuration files for training, sampling, etc.
├── dock_file_save/    # Auto-generated docking results (timestamped folders)
├── evaluation_dock/   # Docking evaluation pipeline (PDBQT conversion, QVina, analysis scripts)
├── figure/            # Images and visual assets
├── generate/          # Utility scripts for the molecular sampling process
├── metrics/           # Evaluation metric toolkits (e.g., SA score)
├── model/             # Core architecture (Encoder, Decoder, and Fusion modules)
└── save/              # Saved model checkpoints, parameters, and sampling data
```



## 📝 Citation
If you find our work helpful in your research, please cite our paper:
```bib
@inproceedings{chen2026tsmgen,
  title={TSMGen: Target-Specific Molecule Generation via Higher-Order Structural Dependencies and Context-Aware Bidirectional Fusion},
  author={Chen, Yaoyu and Lin, Xiaoli and Gong, Ziyi and Pang, Jun},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2026}
}
```
