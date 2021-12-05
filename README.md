# eeg-missing-data

Classification of EEG signals (P300 Event related potential) with missing values.

Code of the paper "Riemannian classification of EEG signals with missing values" (under review).

Link of the paper: https://arxiv.org/abs/2110.10011

To make use of this code, first follow these steps:

1. Install Anaconda: https://docs.anaconda.com/anaconda/install/index.html
2. Create a new environment from the .yml file:

		conda env create -f environment.yml

3. Activate the new environment (with `myenv` replaced by the name of your environment):

		conda activate myenv

4. Make sure the environment is correctly installed:

		conda env list

To reproduce the classification accuracy results (figures 2 and 3 of the paper):

1. Go to the `simulations/` folder, then run:

		python -m pdb demo_P300_dataset.py

Let it run (can take a while). Results will be stored in a file called `accuracies.pkl`.
If time is your enemy, a file of the same name is already there.

2. To plot the results, go to `utils/`, then run:

		python plot.py
