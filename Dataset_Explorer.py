#coding=utf-8

import sys
import os
import math
import pickle
import numpy as np
import pandas
import tensorflow as tf
import scipy
import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import FunctionalGroups
from rdkit import DataStructs

from MOD_CompositeGNN import *
from MOD_CompositeGNN.CompositeGNN import *
from MOD_CompositeGNN import GNN_utils
from MOD_CompositeGNN.MLP import *

#network parameters
EPOCHS = 2000               #number of training epochs
STATE_DIM = 50				#node state dimension
HIDDEN_UNITS_OUT_NET = 200	#number of hidden units in the output network
LR = 0.0001					#learning rate
MAX_ITER = 6				#maximum number of state convergence iterations
VALIDATION_INTERVAL = 10	#interval between two validation checks, in training epochs
TRAINING_BATCHES = 10       #number of batches in which the training set should be split
DROPOUT_RATE = 0.0			#dropout rate for MLPs
AGGREGATION = "average"     #can either be "average" or "sum"
ACTIVATION = "tanh"			#activation function
SIDE_EFFECT_COUNT = 360		#number of side-effects to take into account 
LABEL_DIM = [135, 140, SIDE_EFFECT_COUNT]


#gpu parameters
use_gpu = True
target_gpu = "1"

#script parameters
compound_id = sys.argv[1]
path_data = "Datasets/Nuovo/Output/Soglia_100/"
path_results = "Results/Nuovo/LinkPredictor/coso.txt"
tanimoto_threshold = 0.7
feature_fingerprint_size = 128
tanimoto_fingerprint_size = 2048
ontology_size = 113
splitting_seed = 3
validation_share = 0.1
test_share = 0.1
atomic_number = { 'Li':3, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Mg':12, 'Al':13, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Fe':26, 'Co':27, 'As':33, 'Br':35, 'I':53, 'Au':79 }
atomic_label = { 3:'Li', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 12:'Mg', 13:'Al', 15:'P', 16:'S', 17:'Cl', 19:'K' ,20:'Ca', 26:'Fe', 27:'Co', 33:'As', 35:'Br', 53:'I', 79:'Au' }
label_translator = {'C':1, 'N':2, 'O':3, 'S':4, 'F':5, 'P':6, 'Cl':7, 'I':7, 'Br':7, 'Ca':8, 'Mg':8, 'K':8, 'Li':8, 'Co':8, 'As':8, 'B':8, 'Al':8, 'Au':8, 'Fe':8}
chromosome_dict = {'MT':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20, '21':21, '22':22, 'X':23, 'Y':24}

#function that adjusts NaN features
def CheckedFeature(feature):
	if feature is None:
		return 0.0
	if np.isnan(feature):
		return 0.0
	return feature

#function that binarizes a fingerprint
def BinarizedFingerprint(fp):
	bfp = [0 for i in range(len(fp))]
	for i in range(len(fp)):
		if fp[i] > 0:
			bfp[i] = 1
	return bfp

#load side-effect data
print("Loading side-effects")
in_file = open(path_data+"side_effects.pkl", 'rb')
side_effects = pickle.load(in_file)
in_file.close()
#load gene data
print("Loading genes")
in_file = open(path_data+"genes.pkl", 'rb')
genes = pickle.load(in_file)
in_file.close()
#load drug data
print("Loading drugs")
in_file = open(path_data+"drugs.pkl", 'rb')
drugs_pre = pickle.load(in_file)
in_file.close()
#load drug-side effect links
print("Loading drug - side-effects associations")
in_file = open(path_data+"drug_side_effect_links.pkl", 'rb')
links_dse = pickle.load(in_file)
in_file.close()
#load gene-gene links
print("Loading protein-protein interactions")
in_file = open(path_data+"gene_gene_links.pkl", 'rb')
links_gg = pickle.load(in_file)
in_file.close()
#load drug-gene links
print("Loading drug-gene links")
in_file = open(path_data+"drug_gene_links.pkl", 'rb')
links_dg = pickle.load(in_file)
in_file.close()
#load drug features
print("Loading drug features")
pubchem_data = pandas.read_csv(path_data+"pubchem_output.csv")
#load gene features
print("Loading gene features")
in_file = open(path_data+"gene_features.pkl", 'rb')
gene_data = pickle.load(in_file)
in_file.close()
#load gene ontology molecular function classification
print("Loading molecular function ontology data")
ontology = pandas.read_table(path_data+"david_output.tsv")

#preprocess drug ids
print("Preprocessing drug identifiers")
drugs = list()
for i in range(len(drugs_pre)):
	drugs.append(str(int(drugs_pre[i][4:])))

#determine graph dimensions
print("Calculating graph dimensions")
CLASSES = len(side_effects)		#number of outputs
n_nodes = len(drugs)+len(genes)
n_edges = 2*links_dg.shape[0]+2*links_gg.shape[0]
dim_node_label = max(LABEL_DIM)
type_mask = np.zeros((n_nodes,3), dtype=int)
#build id -> node number mappings
node_number = dict()
inverted_node_number = dict()
for i in range(len(drugs)):
	node_number[str(drugs[i])] = i
	inverted_node_number[i] = str(drugs[i]) 
	type_mask[i][0] = 1
for i in range(len(genes)):
	node_number[str(genes[i])] = i + len(drugs)
	inverted_node_number[i+len(drugs)] = str(genes[i])
	type_mask[i + len(drugs)][1] = 1
#build id -> class number mappings
class_number = dict()
for i in range(len(side_effects)):
	class_number[side_effects[i]] = i

#build output mask
print("Building output mask")
output_mask = np.concatenate((np.ones(len(drugs)), np.zeros(len(genes))))

#build list of positive examples
print("Building list of positive examples")
positive_dsa_list = list()
for i in range(links_dse.shape[0]):
	if str(int(links_dse[i][0][4:])) in node_number.keys():
		#skip side-effects which were filtered out of the dataset
		if links_dse[i][2] not in side_effects:
			continue
		positive_dsa_list.append((node_number[str(int(links_dse[i][0][4:]))],class_number[links_dse[i][2]]))
	else:
		sys.exit("ERROR: drug-side-effect link pointing to incorrect drug id")

#build node feature matrix
print("Building node feature matrix")
nodes = np.zeros((n_nodes, dim_node_label))
#build drug features
for i in pubchem_data.index:
	#skip drugs which were filtered out of the dataset
	if str(pubchem_data.at[i,'cid']) not in node_number.keys():
		continue
	nn = node_number[str(pubchem_data.at[i,'cid'])]
	nodes[nn][0] = CheckedFeature(float(pubchem_data.at[i,'mw']))#molecular weight
	nodes[nn][1] = CheckedFeature(float(pubchem_data.at[i,'polararea']))#polar area
	nodes[nn][2] = CheckedFeature(float(pubchem_data.at[i,'xlogp']))#log octanal/water partition coefficient
	nodes[nn][3] = CheckedFeature(float(pubchem_data.at[i,'heavycnt']))#heavy atom count
	nodes[nn][4] = CheckedFeature(float(pubchem_data.at[i,'hbonddonor']))#hydrogen bond donors
	nodes[nn][5] = CheckedFeature(float(pubchem_data.at[i,'hbondacc']))#hydrogen bond acceptors
	nodes[nn][6] = CheckedFeature(float(pubchem_data.at[i,'rotbonds']))#number of rotatable bonds

#normalize drug features
print("Normalizing drug features")
for i in range(dim_node_label):
	col_min = None
	col_max = None
	for j in range(n_nodes):
		#skip zeros
		if nodes[j][i] == 0:
			continue
		if col_min is None:
			col_min = nodes[j][i]
		if col_max is None:
			col_max = nodes[j][i]
		if nodes[j][i] < col_min:
			col_min = nodes[j][i]
		if nodes[j][i] > col_max:
			col_max = nodes[j][i]
	#do not normalize zero columns
	if col_min is None or col_max is None:
		continue
	for j in range(nodes.shape[0]):
		#do not normalize zeros
		if nodes[j][i] == 0:
			continue
		nodes[j][i] = float(nodes[j][i] - col_min) / float(col_max - col_min)

#build dict of molecular structures
molecule_dict = dict()
for i in pubchem_data.index:
	#skip drugs which were filtered out of the dataset
	if str(pubchem_data.at[i,'cid']) not in node_number.keys():
		continue
	nn = node_number[str(pubchem_data.at[i,'cid'])]
	molecule_dict[nn] = rdkit.Chem.MolFromSmiles(pubchem_data.at[i,'isosmiles'])
#build dicts of fingerprints
feature_fingerprint_dict = dict()
tanimoto_fingerprint_dict = dict()
for k in molecule_dict.keys():
	feature_fingerprint_dict[k] = Chem.RDKFingerprint(molecule_dict[k], fpSize=feature_fingerprint_size)
	tanimoto_fingerprint_dict[k] = Chem.RDKFingerprint(molecule_dict[k], fpSize=tanimoto_fingerprint_size)

#add fingerprints to drug node features
for i in pubchem_data.index:
	#skip drugs which were filtered out of the dataset
	if str(pubchem_data.at[i,'cid']) not in node_number.keys():
		continue
	nn = node_number[str(pubchem_data.at[i,'cid'])]
	#get fingerprint from dictionary and convert it to numpy array
	fingerprint = np.array((1,))
	rdkit.DataStructs.cDataStructs.ConvertToNumpyArray(feature_fingerprint_dict[nn], fingerprint)
	#add fingerprint
	nodes[nn][-feature_fingerprint_size:] = fingerprint

#build list of drug-drug connections on the basis of chemical fingerprint similarity
ddc_list = list()
for i in range(len(drugs)):
	for j in range(len(drugs)):
		if i == j:
			continue
		tanimoto_coeff = DataStructs.TanimotoSimilarity(tanimoto_fingerprint_dict[node_number[drugs[i]]],tanimoto_fingerprint_dict[node_number[drugs[j]]])
		if tanimoto_coeff >= tanimoto_threshold:
			ddc_list.append([node_number[drugs[i]], node_number[drugs[j]]])
print("Adding "+str(len(ddc_list))+" drug-drug edges based on Tanimoto similarity ( coeff >= "+str(tanimoto_threshold)+" )")
n_edges = n_edges + len(ddc_list)

#build gene features
print("Adding gene features")
for i in range(gene_data.shape[0]):
	#skip genes which were filtered out of the dataset
	if gene_data[i,0] not in node_number.keys():
		continue
	nn = node_number[gene_data[i,0]]
	nodes[nn][0] = float(gene_data[i,1])#dna strand (-1 or +1)
	nodes[nn][1] = float(gene_data[i,2])#percent GC content (real value in [0,1])
	nodes[nn][2+chromosome_dict[gene_data[i,4]]] = float(1)#one-hot encoding of chromosome

#add ontology features
start = 27
for i in ontology.index:
	#collect list of genes associated to i-th term
	gene_list = ontology.at[i,"Genes"].split(", ")
	#add a "1" to each gene in the list in the (i+start)-th column
	for g in gene_list:
		#skip void strings (a by-product of splitting)
		if g == "": continue
		nn = node_number[g]
		nodes[nn][start+i] = 1

#build target tensor
print("Building target tensor")
targets = np.zeros((len(drugs),len(side_effects)))
for p in positive_dsa_list:	
	targets[p[0]][p[1]] = 1

#select most common side effects
se_counts = np.sum(targets, axis=0)
sorted_counts = np.sort(-se_counts)
se_count_threshold = -sorted_counts[SIDE_EFFECT_COUNT-1]
del_indices = list()
for i in range(targets.shape[1]):
	if se_counts[i] < se_count_threshold:
		del_indices.append(i)
for i in range(targets.shape[1]):
	if se_counts[i] == se_count_threshold and len(del_indices) < CLASSES - SIDE_EFFECT_COUNT:
		del_indices.append(i)
targets = np.delete(targets, del_indices, axis=1)
CLASSES = SIDE_EFFECT_COUNT
if targets.shape[1] != CLASSES:
	sys.exit("ERROR: Error while selecting most common side-effect")

#build arcs tensor
print("Building arc tensor")
arcs = np.zeros((n_edges,2), dtype=int)
l = 0
#add drug-gene edges
for i in range(links_dg.shape[0]): 
	arcs[l][:] = [node_number[str(int(links_dg[i][0]))],node_number[str(links_dg[i][1])]]
	arcs[l+1][:] = [node_number[str(links_dg[i][1])],node_number[str(int(links_dg[i][0]))]]
	l = l+2
#add gene-gene edges
for i in range(links_gg.shape[0]):
	arcs[l][:] = [node_number[str(links_gg[i][0])],node_number[str(links_gg[i][1])]]
	arcs[l+1][:] = [node_number[str(links_gg[i][1])],node_number[str(links_gg[i][0])]]
	l = l+2
#add drug-drug edges
for ddc in ddc_list:
	arcs[l][:] = ddc
	l = l+1
arcs = np.array(arcs)

### DEBUG START ###
### DEBUG: calculate graph diameter
'''
print("Calculating graph diameter")
import networkx as nx
g = nx.Graph()
for i in range(len(nodes)):
	g.add_node(i)
for i in range(len(arcs)):
	g.add_edge(arcs[i][0], arcs[i][1])
diameter = nx.algorithms.distance_measures.diameter(g)
print("Graph Diameter = "+str(diameter))
sys.exit()
'''
### DEBUG STOP ###

### DEBUG START ###
### DEBUG: print final list of genes
'''
print("Printing final list of genes")
out_file = open("gene_ids.txt", 'w')
for g in genes:
	out_file.write(g+"\n")
out_file.close()
sys.exit()
'''
### DEBUG STOP ###

#split the dataset
print("Splitting the dataset")
validation_size = int(validation_share*targets.shape[0])
test_size = int(test_share*targets.shape[0])
index = np.array(list(range(targets.shape[0])))
np.random.seed(splitting_seed)
np.random.shuffle(index)
test_index = index[:test_size]
validation_index = index[test_size:test_size+validation_size]
training_index = index[test_size+validation_size:]
#build set masks
te_mask = np.zeros(targets.shape[0], dtype=int)
va_mask = np.zeros(targets.shape[0], dtype=int)
tr_mask = np.zeros(targets.shape[0], dtype=int)
for i in test_index:
	te_mask[i] = 1
for i in validation_index:
	va_mask[i] = 1
for i in training_index:
	tr_mask[i] = 1
#concatenate all-zero set mask extensions for gene nodes
te_mask = np.concatenate((te_mask,np.zeros(len(genes), dtype=int)))
va_mask = np.concatenate((va_mask,np.zeros(len(genes), dtype=int)))
tr_mask = np.concatenate((tr_mask,np.zeros(len(genes), dtype=int)))

### DEBUG START ###
'''
print("Printing dimensions of dataset components")
print(nodes.shape)
print(arcs.shape)
print(expanded_targets.shape)
print(tr_mask.shape)
print(va_mask.shape)
print(te_mask.shape)
print(type_mask.shape)
sys.exit()
'''
### DEBUG STOP ###

#explore the dataset and retrieve input compound information
nn = node_number[compound_id]
#obtain list of neighbors
neighbors = list()
for a in arcs:
	if nn == a[0]: #just check left side, as all connections are bidirectional (two arcs exist between each pair of connected nodes)
		neighbors.append(inverted_node_number[a[1]]) 
#obtain targets
target_list = list()
for i in range(SIDE_EFFECT_COUNT):
	if targets[nn][i] == 1:
		target_list.append(side_effects[i])
#print information
print("************************")
print("Neighbor nodes:")
print(neighbors)
print("************************")
print("Side effects:")
print(target_list)
print("************************")
