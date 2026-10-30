# =============================================================================
# FragProtac
# This model is designed for binary activity prediction of PROTAC molecules by modeling
# the triple interactions among warhead, linker and E3 ligand, together with pathway embedding.
# This repository can help reproduce the experiments of our work.
# Work in progress. Citation will be added upon acceptance.
#
# Training:
# 1. Prepare the data.
#    This script requires warhead SMILES, linker SMILES, E3 ligand SMILES,
#    precomputed pathway embedding and binary activity label.
#    All information should be organized in one CSV file.
#    We provide the format of the CSV in the description below.
#    If you want to use your own dataset, please modify the file path in the script.
#
# 2. Check CSV format.
#    The input CSV file must contain these columns:
#    label, warhead smiles, linker smiles, e3 ligand smiles, Pathway_Combined_128d
#    Missing SMILES will be automatically replaced with C for robustness;
#    missing pathway vectors will be padded with zero vectors.
#
# 3. Prepare the environment.
#    You can build the conda environment with the following commands.
#    We train and test this model on NVIDIA GPU with CUDA support.
#
#        conda create -n fragprotac python=3.9
#        conda activate fragprotac
#        pip install torch torchvision torchaudio
#        pip install torch_geometric
#        pip install rdkit-pypi pandas numpy scikit-learn matplotlib seaborn
#
#    We recommend matching the PyTorch and PyG versions with your CUDA driver.
#
# 4. Modify configuration.
#    Open the training script and adjust the global paths and hyperparameters as needed:
#        CSV_PATH = "/FragProtac/data/PROTAC-fine/protac-fine.csv"
#        OUTPUT_DIR = "/FragProtac"
#
# 5. Run the training script.
#        python FragProtac_train.py
# =============================================================================
