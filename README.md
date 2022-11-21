# MolecularGNN
The MolecularGNN model is a classifier is based on Graph Neural Networks, a connectionist model capable of processing data in form of graphs. This is an improvement over a previous method, called DruGNN[2], as it is also capable of extracting information from the graph--based molecular structures, producing a task--based neural fingerprint (NF) of the molecule which is adapted to the specific task.

# Use
This repository contained 3 training scripts to trained Deep Neural Network classifier on Drug Side Effect Prediction (DSE):

1. GNN_node_classifier.py which train a DrugGNN model as described in [2]. 
2. GNN_molecule_classifier.py which train a DSE classifier base only on our neural figerprint.
3. GNN_MinN_classifier.py which train the full MolecularGNN model.

All script can take an optionnal command line argument run_id to differentiate training from one another. All parameters related to leraning must be modify inside of the corresponding training script.

# Credit
This work make use of code from:

1. Niccolò Pancino, Pietro Bongini, Franco Scarselli, Monica Bianchini,
  GNNkeras: A Keras-based library for Graph Neural Networks and homogeneous and heterogeneous graph processing,
  SoftwareX, Volume 18, 2022, 101061, ISSN 2352-7110, https://doi.org/10.1016/j.softx.2022.101061. Accessible at https://github.com/NickDrake117/GNNkeras.
2. Bongini, Pietro & Scarselli, Franco & Bianchini, Monica & Dimitri, Giovanna & Pancino, Niccolò & Lio, Pietro. (2022).
  Modular multi-source prediction of drug side-effects with DruGNN.
  IEEE/ACM transactions on computational biology and bioinformatics PP (2022): n. pag.
