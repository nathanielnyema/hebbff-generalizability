# Generalization in the Idealized HebbFF Network
## Author: Nathaniel Nyema

This folder contains all of the code used to generate the analyses for my final project for CNS187 based on the paper Meta-learning synaptic plasticity and memory addressing for continual familiarity detection by [Tyulmankov et al](https://www.sciencedirect.com/science/article/pii/S0896627321009478?via%3Dihub#bib3). The notebook `analyses.ipynb` contains the analyses themselves and it imports some custom helper functions from utils.py. To run the code I recommend creating a conda environment using the environment.yml file as well as creating jupyter kernel to run the notebook in. Some code to do this could be as follows:

```
conda env create -f environment.yml
conda activate hebbff
python -m ipykernel install --user --name=hebbff
```