#coding=utf-8


import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score

from GNN.Models.MLP import MLP
from GNN.Models.GNN import GNNgraphBased
from GNN.Models.CompositeGNN import CompositeGNNnodeBased
from GNN.graph_class import GraphObject
from GNN.Sequencers.GraphSequencers import MultiGraphSequencer, CompositeMultiGraphSequencer, CompositeSingleGraphSequencer

from GNN_molecule_classifier import load_molecules, ARC_DIM, LABEL_DIM
from GNN_node_classifier import load_drugs, LABEL_DIMS
from GNN_MinN_utils import MinN_model, MinN_Sequence, weight
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

name="baseline"
#network parameters
EPOCHS = 500               #number of training epochs
LR = 0.001					#learning rate
AGGREGATION = "average"     #can either be "average" or "sum"
ACTIVATION = "tanh"			#activation function
SIDE_EFFECT_COUNT = 280		#number of side-effects to take into account
INNER_DIM = 380

#gpu parameters
use_gpu = True
target_gpu = "1"

#molecules embedding graphs and submodel
graphs = load_molecules()

MG_Sequencer = MultiGraphSequencer(graphs, 'g', AGGREGATION, 32000, shuffle = False)

#drug-effect graphs
DG_Trs, DG_Va, DG_Te = load_drugs(num_batch = 10,molecular_feature_size = INNER_DIM)
CLASSES = DG_Te.targets.shape[1]

DG_Tr_Sequencer = CompositeMultiGraphSequencer(DG_Trs, 'n', AGGREGATION, 1, shuffle=False)
DG_Va_Sequencer = CompositeSingleGraphSequencer(DG_Va, 'n', AGGREGATION, 9999) 
DG_Te_Sequencer = CompositeSingleGraphSequencer(DG_Te, 'n', AGGREGATION, 9999)

def create_model():
    #molecules embedding submodel
    M_nb_layers = 3
    M_nb_layers_out = 2
    M_batch_norm = True
    M_L2 = 0.001
    M_dropout = 0.0
    M_activation = 'relu'
    M_state_dim = 150
    M_max_iter = 4

    M_netSt = MLP(input_dim=(2*M_state_dim+2*LABEL_DIM+ARC_DIM,),
                        layers=[M_state_dim for i in range(M_nb_layers)],
                        activations=[M_activation for i in range(M_nb_layers)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers)],
                        kernel_regularizer=[tf.keras.regularizers.L2(M_L2) for i in range(M_nb_layers)],
                        bias_regularizer=[tf.keras.regularizers.L2(M_L2) for i in range(M_nb_layers)],
                        dropout_rate=M_dropout, dropout_pos=1,batch_normalization=M_batch_norm)

    M_netOut = MLP(input_dim=(M_state_dim+LABEL_DIM,),
                        layers=[INNER_DIM for i in range(M_nb_layers_out)],
                        activations=[M_activation for i in range(M_nb_layers_out)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers_out)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers_out)],
                        kernel_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(M_nb_layers_out)],
                        bias_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(M_nb_layers_out)],
                        dropout_rate=M_dropout, dropout_pos=1,batch_normalization=M_batch_norm)

    moleculesGNN = GNNgraphBased(M_netSt, M_netOut, M_state_dim, M_max_iter, 0.01)


    #drug-effect submodel
    N_nb_layers = 2
    N_batch_norm = True
    N_L2 = 0.0
    N_dropout = 0.1
    N_activation = 'relu'
    N_state_dim =50
    N_max_iter = 4
    HIDDEN_UNITS_OUT_NET = 200

    netSt_drugs = MLP(input_dim=(2*N_state_dim+LABEL_DIMS[0]+sum(LABEL_DIMS),),
                        layers=[N_state_dim for i in range(N_nb_layers)],
                        activations=[N_activation for i in range(N_nb_layers)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        kernel_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        bias_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        dropout_rate=N_dropout, dropout_pos=1,batch_normalization=N_batch_norm)
    netSt_drugs_augmented = MLP(input_dim=(2*N_state_dim+LABEL_DIMS[2]+sum(LABEL_DIMS),), 
                        layers=[N_state_dim for i in range(N_nb_layers)],
                        activations=[N_activation for i in range(N_nb_layers)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        kernel_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        bias_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        dropout_rate=N_dropout, dropout_pos=1,batch_normalization=N_batch_norm)
    netSt_genes = MLP(input_dim=(2*N_state_dim+LABEL_DIMS[1]+sum(LABEL_DIMS),),
                        layers=[N_state_dim for i in range(N_nb_layers)],
                        activations=[N_activation for i in range(N_nb_layers)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        kernel_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        bias_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        dropout_rate=N_dropout, dropout_pos=1,batch_normalization=N_batch_norm)
    netOut = MLP(input_dim=(N_state_dim+LABEL_DIMS[0],), layers=[HIDDEN_UNITS_OUT_NET,CLASSES], activations=[N_activation, 'sigmoid'],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                        kernel_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(2)],
                        bias_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(2)],
                        dropout_rate=N_dropout, dropout_pos=1,batch_normalization=N_batch_norm)

    dseGNN = CompositeGNNnodeBased([netSt_drugs, netSt_genes, netSt_drugs_augmented], netOut, N_state_dim, N_max_iter, 0.001)

    gamma = 5
    mu = 0.5

    #calculate class weight
    class_weight = weight(DG_Tr_Sequencer.targets, mu=0.0)

    #define loss function with weight
    BC_object = tf.keras.losses.BinaryFocalCrossentropy(gamma = gamma, reduction = 'none')
    def loss(y_true, y_pred):
        l = BC_object(y_true[..., tf.newaxis], y_pred[..., tf.newaxis])
        l = tf.math.multiply(l, class_weight)
        return tf.reduce_mean(l)

    #create model
    Model = MinN_model(moleculesGNN, dseGNN, embedding_start=LABEL_DIMS[0]-INNER_DIM,embedding_size=INNER_DIM)
    Model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                loss=loss,
                average_st_grads=False,
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                F1Score(num_classes=SIDE_EFFECT_COUNT, threshold=0.5, average='micro'),
                tf.keras.metrics.AUC(num_thresholds=200,curve='ROC',multi_label=True),
                tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),
                tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.PrecisionAtRecall(0.9)],
                run_eagerly=True)
    
    return Model

#create data sequences
Tr_sequencer = MinN_Sequence(MG_Sequencer,DG_Tr_Sequencer)
Va_sequencer = MinN_Sequence(MG_Sequencer,DG_Va_Sequencer)
Te_sequencer = MinN_Sequence(MG_Sequencer,DG_Te_Sequencer)


for i in range(10):
    print("-------------",i,"-------------")

    Model = create_model()
    #create callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='default/', histogram_freq=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max', patience=50, restore_best_weights=True)

    #train
    Model.fit(Tr_sequencer, epochs=EPOCHS, validation_data =  Va_sequencer, callbacks=[tensorboard_callback,earlystop], verbose = 0)

    #evaluate
    Model.evaluate(Tr_sequencer)
    Model.evaluate(Va_sequencer)
    Model.evaluate(Te_sequencer)
    result = Model.predict(Te_sequencer)
    np.savez_compressed("result/"+name+"/"+str(i)+".npz", result=result)
