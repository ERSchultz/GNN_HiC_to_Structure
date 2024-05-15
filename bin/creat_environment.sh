envName=GNN_pytorch
conda create --name $envName -y
conda activate $envName
conda install -y python=3.9 pytorch=1.9 pyg pytorch-scatter torchvision cudatoolkit matplotlib imageio numpy jupyterlab pillow seaborn numba pandas scikit-learn scikit-image scipy pybigwig pybind11 sympy isort -c pytorch -c conda-forge -c bioconda -c pyg
pip install pynvml importmagic hic-straw hicrep opencv-python bioframe
conda deactivate
