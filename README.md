# curvature_penalization
Curvature penalization for Auto Encoders

This package is the ultimate version of code used in Alexey Lazarev's PhD thesis. The core of the project is curvature regularization technique and Neural Riemannian clustering.

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- ricci_regularization: Core of the package. 
|  |-- Architectures.py : Choice of AE architectures: TorusAE, TorusConvAE, etc.
|  |-- DataLoaders.py : Dataset loading and neural net weights loading.
|  |-- FiniteDifferences.py : Computing all useful Riemannian geometry tensors via Finite differences.
|  |-- Losscomputation.py : Train,test and loss comutation functions.
|  |-- OODTools.py : functions for OOD sampling
|  |-- PlottingTools.py : Plotting in latent space: Manifold plots, heatmaps, etc..
|  |-- Ricci.py : Computing all useful Riemannian geometry tensors via Autograd
|  |-- RiemannianKmeansTools.py : Ingredients of K-means and its plotting.
|  |-- Schauder.py : Schauder basis.
|  |-- SyntheticGaussians.py : Creating the Synthetic Gaussians dataset


|-- ipynb: Contains Python notebooks which demonstrate how the code works. Most important files:
|  |-- AE_torus_training.ipynb: training of the AE 
|  |-- AE_torus_report.ipynb: building the report

|-- Benchmarks: TODO
|-- README.md: This file
```

## A quick start:

Step 1:
Train the neural net with curvature regularization by launching ipynb/AE_torus_training.ipynb. Save weights.

Step 2:
Input: Results of Step 1: weigts of pre-trained autoencoder.
Generate and visualize the report of the training. Launch ipynb/AE_torus_report.ipynb.

Step 3:
Input: Results of Step 1: weigts of pre-trained autoencoder.
Check the geodesics in the latent space. Launch ipynb/Benchmarks/Geodesic_grids.ipynb.

Step 4:
Input: Results of Step 1: weigts of pre-trained autoencoder.
Check the quasi-isometric embedding of the Torus latent space. Launch ipynb/Benchmarks/torus3dembedding_training.ipynb 

Step 5:
Input: Results of Step 4.
Visualize torus 3d embedding with ipynb/Benchmarks/torus3dembedding_visualization.ipynb

Step 6:
Input: Results of Step 1: weigts of pre-trained autoencoder.
Perform clustering. Launch ipynb/Benchmarks/Chapter_5_multiple_RiemannianK-means_execution.ipynb.

Step 7:
Input: Results of Step 5: points selected for clustering, labels assigned by Riemannian K-means. 
Visualize clustering results with ipynb/Visualization/Chapter_5_Octopus_plotting.ipynb. Perform Euclidean clustering, compute F-scores and Voronoi's cells with ipynb/Visualization/Chapter_5_Clustering_report.ipynb

## Installation

1. Create new virtual environment

```bash
python3 -m venv .venv_ricci
```

(Do
sudo apt install python3-venv
if needed)

2. Activate virtual environment

```bash
source .venv_ricci/bin/activate
```

3. Upgrade pip, wheel and setuptools 

```bash
pip install --upgrade pip
pip install --upgrade setuptools
pip install wheel
```

4. Install PyTorch with CUDA supprt

```bash
pip install torch torchvision torchaudio
```

5. In order to use Jupyter with this virtual environment .venv
```bash
pip install ipykernel
python -m ipykernel install --user --name=.venv_ricci
```
(see https://janakiev.com/blog/jupyter-virtual-envs/ for details)

6. Install the `ricci_regularization` package.

```bash
python setup.py develop
```

7. Additionnal packages which could be removed in ulterior versions
```bash
pip install stochman geomstats
```

## Configuration
Nothing to do

## Credits
Later
