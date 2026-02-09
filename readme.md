# Deformable 3D Point Cloud Perturbations using Cage-based Deformation for Semantic Consistency

## Installation
```bash
git clone https://github.com/kslee990614/Deformable-3D-Point-Cloud-Perturbations-using-Cage-based-Deformation-for-Semantic-Consistency.git
# install dependency
cd pytorch_points
conda env create --name pytorch-all --file environment.yml
python setup.py develop
# install pymesh2
# if this step fails, try to install pymesh from source as instructed here
# https://pymesh.readthedocs.io/en/latest/installation.html
# make sure that the cmake 3.15+ is used
pip install pymesh/pymesh2-0.2.1-cp37-cp37m-linux_x86_64.whl
# install other dependecies
pip install -r requirements.txt
```
## Trained model
Download trained models from https://igl.ethz.ch/projects/neural-cage/trained_models.zip.

Unzip under `trained_models`. You should get several subfolders under `trained_models`, e.g. `trained_models/chair_ablation_full` etc.

### Optional
install Thea https://github.com/sidch/Thea to batch render outputs

