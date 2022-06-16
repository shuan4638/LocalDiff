# LocalDiff
## A LocalRetro-like DNN for activation energy prediction for AMSG member training
Implementation of [Deep Learning of Activation Energies, J. Phys. Chem. Lett. 2020](https://pubs.acs.org/doi/10.1021/acs.jpclett.0c00500) with dgl python package.<br>
The data is downloaded from https://zenodo.org/record/3715478, published at [Reactants, products, and transition states of elementary chemical reactions based on quantum chemistry, Sci Data, 2020](https://www.nature.com/articles/s41597-020-0460-4)

## Developer
Shuan Chen (contact: shuankaist@kaist.ac.kr)<br>

## Requirements
* Python (version >= 3.6) 
* Numpy (version >= 1.16.4) 
* PyTorch (version >= 1.0.0) 
* RDKit (version >= 2019)
* DGL (version >= 0.5.2)
* DGLLife (version >= 0.2.6)

## Requirements
Create a virtual environment to run the code of LocalDiff.<br>
Install pytorch with the cuda version that fits your device.<br>
```
git clone https://github.com/shuan4638/LocalDiff.git
cd LocalDiff
conda create -c conda-forge -n rdenv python=3.7 -y
conda activate rdenv
conda install pytorch cudatoolkit=10.2 -c pytorch -y
conda install -c conda-forge rdkit -y
pip install dgl
pip install dgllife
```


## Usage

### [1] Train LocalDiff model
Go to the script folder
```
cd ../scripts
```
and run the following to train the model
```
python Train.py
```
The trained model will be saved at ` LocalDiff/models/LocalDiff.pth`<br>

### [2] Test LocalRetro model
See `demo.ipynb`

## Objective
According to the original paper, the mean absolute error (MAE) is 1.7 ± 0.1 kcal mol–1 and the root-mean-square error (RMSE) is 3.4 ± 0.3 kcal mol–1<br>
Try to perform better!
