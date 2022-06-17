# LocalDiff
Implementation of [Deep Learning of Activation Energies, J. Phys. Chem. Lett. 2020](https://pubs.acs.org/doi/10.1021/acs.jpclett.0c00500) with dgl python package.<br><br>
<img src="https://pubs.acs.org/cms/10.1021/acs.jpclett.0c00500/asset/images/large/jz0c00500_0001.jpeg" height="400">


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

## Data
The data is downloaded from https://zenodo.org/record/3715478, published at [Reactants, products, and transition states of elementary chemical reactions based on quantum chemistry, Sci Data, 2020](https://www.nature.com/articles/s41597-020-0460-4)
I randomly splitted the wb97xd3.csv to train/val/test and hide the test set :p.


## Usage

### [1] Train LocalDiff model
Go to the `scripts` folder and run the following to train the model
```
python Train.py
```
The trained model will be saved at ` LocalDiff/models/LocalDiff.pth`<br>

### [2] Demo the trained model on individual reaction
See `demo.ipynb`

## Objective
According to the original paper, the mean absolute error (MAE) is 1.7 ± 0.1 kcal/mol and the root-mean-square error (RMSE) is 3.4 ± 0.3 kcal/mol<br>
The val RMSE of this repo is ~16 kcal/mol<br>
Try to beat the baseline!
