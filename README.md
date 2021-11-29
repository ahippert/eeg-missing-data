# eeg-missing-data

CODE IN REFACTORING PHASE.

Classification of EEG signals (P300 Event related potential) with missing values.

Code of the paper "Riemannian classification of EEG signals with missing values" (under review).

To launch this code, follow these steps:

1. Install Anaconda.
2. Create a new environment from the .yml file:

		conda env create -f environment.yml

3. Activate the new environment (with myenv replaced by the name of your environment):

		conda activate myenv

4. Make sure the environment is correctly installed:

		conda env list

5. Then run (results will be stored in a .pkl file):

		python -m pdb classification_example.py

6. Plot the results:

		python plot.py

7. Compute p-values:

		python statistical_test.py
