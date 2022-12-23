import NoN_EEG_utils
from NoN_EEG_utils import Subject # if you don't specifically import Subject you can't load the data

import numpy as np
import os
from pathlib import Path
import re
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from scipy.stats import zscore
import networkx as nx
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans



# -------------------------------------------------- Initialize parameters
data_exist = 1 # Zero if you want to load the raw brain connectome data from the server.
results_subjects_exist = 1 # zero if you want to re-run the MANE algorithm on all subjects. (Takes ~1min/subject; total: ~30min)
Corrs_exist = 1 # zero if you want to re-calculate the correlations between empirical intra-layer FC and MANE embedding-derived intra-layer FC (takes ~30min total) 



# -------------------------------------------------- Load/Create data
if data_exist:
    with open('Sample_data.pickle', 'rb') as f:
        subjects = pickle.load(f)
else:
    # initilize directories
    main_dir = os.getcwd()
    data_dir = Path('Y:/') / 'mostame2' / 'UIUC' / 'fMRI-iEEG' / 'codes' / 'Replication in fMRI-EEG Paris data (Tom)'
    print(data_dir)
    # Define subjects as objects including their dFC data 
    files = os.listdir(data_dir)
    pattern = '^T[\S]*_dFCs_1.mat'
    file_names = [text for text in files if re.findall(pattern, text)]
    subjects = []
    for ind, name in tqdm( enumerate(file_names), desc='Loading subjects data: ' ):
        print('Loading subject ' + str(ind))
        t = time.time()
        file_dir = os.path.join(data_dir, name)
        subjects.append( NoN_EEG_utils.Subject(ind, file_dir) )
        print('Time elapsed for subject #' + str(ind) + ': ' + str(time.time() - t))
    with open('Sample_data.pickle', 'wb') as f:
        pickle.dump(subjects, f)



# -------------------------------------------------- Run MANE algorithm
if results_subjects_exist:
    with open('results_subjects.pickle', 'rb') as f:
        subjects = pickle.load(f)
else:
    for ind, subject in enumerate(subjects):
        print(f'--------------------------------- Subject: {ind + 1} ---------------------------------')
        _ = subject.get_NoN(edge_count_Threshold=0.2, latent_dim=40, Null_Repeat = 100, threshold = 3)
    # save results
    with open('results_subjects.pickle', 'wb') as f:
        pickle.dump(subjects, f)



# -------------------------------------------------- MANE Validation
# --------- Approach 1: Test performance of within-layer FC prediction
if Corrs_exist:
    with open('results_corrs.pickle', 'rb') as f:
        temp = pickle.load(f)
        Corrs = temp[0]
        Corrs_null = temp[1]
else:
    Corrs = [None] * len(subjects)
    Corrs_null = [None] * len(subjects)
    for ind, subject in enumerate(subjects):
        print(f'--------------------------------- Subject: {ind + 1} ---------------------------------')
        Empirical = subject.dFC
        Recovered_NoN = subject.NoN_dFC_latent_space
        Corrs[ind], Corrs_null[ind] = NoN_EEG_utils.performance_within_layer(Empirical, Recovered_NoN)
    Corrs = np.stack(Corrs, axis=0)
    Corrs_null = np.stack(Corrs_null, axis=0)
    # save results
    with open('results_corrs.pickle', 'wb') as f:
        pickle.dump([Corrs, Corrs_null], f)

# plot results        
fig = NoN_EEG_utils.plot_performance_within_layer(np.nanmean(Corrs, axis=0), np.nanmean(Corrs_null, axis=0))
fig.savefig('fig1.png')


# --------- Approach 2: Validate on a manipulated data
subject_manipulated = deepcopy(subjects[0])
t = 0
source_layer = 0
destination_layer = 1
subject_manipulated.dFC[destination_layer, :,:, t] = deepcopy( subject_manipulated.dFC[source_layer, :,:, t] )
_ = subject_manipulated.get_NoN(edge_count_Threshold=0.2, latent_dim=40, Null_Repeat = 100, threshold = 3)
fig, ax = plt.subplots(1,1)
NoN_EEG_utils.plot_NoN_dFC( subject_manipulated.NoN_dFC_latent_space[:,:, t], ax=ax, bin=False)
fig.savefig('Fig2.png')



# -------------------------------------------------- Kmeans analysis
# Pool NoNs across subjects for clustering analysis
NoN_allsubs_alltimes = []
t_max = subjects[0].NoN_dFC_latent_space.shape[-1]
for t in tqdm(range( t_max )):
    NoN_allsubs = []
    for ind, subject in enumerate(subjects):
        FC = deepcopy(subject.NoN_dFC_latent_space[:, :, t])
        FC_vec = NoN_EEG_utils.vectorize_FC(FC) 
        NoN_allsubs.append( FC_vec )
    NoN_allsubs = np.stack(NoN_allsubs, axis=0)
    NoN_allsubs_alltimes.append( NoN_allsubs )
NoN_allsubs_alltimes = np.concatenate(NoN_allsubs_alltimes, axis=0) # (subject * time) * features

# Preprocess/clean the vectorized NoN data
outlier_threshold = 3
imp = SimpleImputer(missing_values=np.nan, strategy='median')
NoN_allsubs_alltimes = imp.fit_transform(NoN_allsubs_alltimes)
NoN_allsubs_alltimes_normed = normalize(NoN_allsubs_alltimes)
scaler = StandardScaler()
NoN_allsubs_alltimes_zscored = scaler.fit_transform(NoN_allsubs_alltimes_normed)
# replace outliers with median in each feature
NoN_allsubs_alltimes_zscored_nooutliers = deepcopy(NoN_allsubs_alltimes_zscored)
NoN_allsubs_alltimes_zscored_nooutliers[ np.absolute(NoN_allsubs_alltimes_zscored_nooutliers) > outlier_threshold ] = outlier_threshold
# show the preprocessed data on carpet plot to assure the quality of data
fig = plt.figure(figsize=(15,5))
plt.imshow(NoN_allsubs_alltimes_zscored_nooutliers[:, :4000], cmap='seismic')
plt.colorbar()
fig.savefig('fig3.png')

# Run Kmeans
N_clusters = 6
kmeans = KMeans(n_clusters=N_clusters, random_state=48).fit(NoN_allsubs_alltimes_zscored)

# plot NoN clusters
fig, ax = plt.subplots(np.ceil(N_clusters / 2).astype(int), 2, figsize=(5, 8))
ax = ax.ravel()
FC_clusters = []
N = subjects[0].NoN_dFC_latent_space.shape[0]
for cluster in range( N_clusters ):
    inds = np.where(kmeans.labels_ == cluster)[0]
    centroid = NoN_allsubs_alltimes[inds, :].mean(axis=0)
    FC = NoN_EEG_utils.de_vectorize_FC(centroid, N)
    np.fill_diagonal(FC, 0)
    FC_clusters.append( zscore(FC, axis=None) > 1 )
    NoN_EEG_utils.plot_NoN_dFC( FC_clusters[cluster], ax=ax[cluster], bin=True)
    ax[cluster].axis(False)
    fig.tight_layout(pad=2)
    ax[cluster].set_title(f'Cluster #{cluster + 1}')
fig.savefig('fig4.png')



# -------------------------------------------------- Extract cross-timescale interaction patterns
N_layers = subjects[0].dFC.shape[0]
N_elecs = subjects[0].dFC.shape[1]
freqpair_strength_allclusters = []
fig, ax = plt.subplots(2, np.ceil(N_clusters /2 ).astype(int), figsize=(6, 12))
ax = ax.ravel()
for cluster_ind, FC_cluster in enumerate(FC_clusters):
    freqpair_strength = np.empty( (N_layers, N_layers) )
    for freq1 in range(N_layers):
        for freq2 in range(N_layers):
            a = freq1 * N_elecs
            b = freq2 * N_elecs
            freqpair_strength[freq1, freq2] = np.triu(FC_cluster[a : a + N_elecs, b : b + N_elecs], k=1).sum()
    freqpair_strength_allclusters.append( freqpair_strength )
    # create the graph        
    G = nx.from_numpy_array(freqpair_strength, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)
    pos = {0: (0, 0), 1: (0, 5), 2: (0, 10), 3: (0, 15), 4: (0, 20)}
    node_dict = {0: 'Delta', 1: 'Theta', 2: 'Alpha', 3: 'Beta', 4: 'Gamma'}
    edge_list = list( G.edges(data=True) )
    edge_width = [e['weight'] for u,v,e in G.edges(data=True)]
    edge_width = np.array(edge_width)
    edge_width = (edge_width - edge_width.min() ) / (edge_width.max() - edge_width.min())
    edge_color = []
    for (u, v, e) in edge_list:
        if u < v:
            clr = [0, 0.5, 0]
        elif u > v:
            clr = [0, 0, 0.8]
        else:
            clr = [1, 0, 0]
        edge_color.append(clr)
    plt.sca( ax[cluster_ind] )
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, width= 1 + 2 * edge_width ** 2, alpha=np.clip(edge_width ** 2, 0.1, 1), connectionstyle="arc3, rad=0.3", edge_color=edge_color, arrowsize=15, min_source_margin=0, min_target_margin=25)
    nx.draw_networkx_nodes(G, pos, node_size=1000, alpha=1, node_color='k')
    nx.draw_networkx_labels(G, pos, labels=node_dict, font_size=8, font_color='w', font_weight='bold')
    ax[cluster_ind].set_aspect('equal')
    ax[cluster_ind].set_title(f'Cluster #{cluster_ind + 1}')
    plt.setp(ax[cluster_ind].spines.values(), linewidth=3)
    plt.xlim(-4, 4)
    plt.ylim(-2, 24)
fig.savefig('fig5.png')



# -------------------------------------------------- Extract state transition probabilities
dynamics_allsubs = kmeans.labels_.reshape(len(subjects), -1)
tp_allsubs = np.empty( (N_clusters, N_clusters, dynamics_allsubs.shape[0]) )
for row in range( dynamics_allsubs.shape[0] ):
    dynamics = dynamics_allsubs[row, :] 
    tp = np.empty((N_clusters, N_clusters))
    for state1 in range( N_clusters ):
        for state2 in range( N_clusters ):
            tp[state1, state2] = ( ( dynamics == state1 ) & np.roll(dynamics == state2, -1) ).sum() / dynamics.size
    tp_allsubs[:, :, row] = tp

# plot results
fig = plt.figure()
_ = plt.imshow( tp_allsubs.mean(axis=-1), cmap='seismic')
_ = plt.xticks(ticks=np.arange(0, N_clusters), labels=np.arange(1, N_clusters + 1))
_ = plt.yticks(ticks=np.arange(0, N_clusters), labels=np.arange(1, N_clusters + 1))
_ = plt.setp(plt.gca().spines.values(), linewidth=3)
_ = plt.colorbar()
plt.title('Transition probability between clusters')
fig.savefig('fig6.png')



# -------------------------------------------------- Extract Fractional Occupancy of the states
FO = [(kmeans.labels_ == ind).sum() / kmeans.labels_.size for ind in range(N_clusters)]
fig = plt.figure()
plt.bar(np.arange(1, N_clusters+1), FO, width=0.9)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylabel('Fractional Occupancy')
plt.xlabel('Clusters')
fig.savefig('fig7.png')



# -------------------------------------------------- Plot the group-average EEG Backbone
Intra_layer_backbone_all = np.zeros( subjects[0].binarize_dFC()[0, :,:,0].shape ) 
for subject in subjects:
    dFC_bin = subject.binarize_dFC()
    Intra_layer_backbone = np.mean(dFC_bin.mean(axis=-1), axis=0)
    Intra_layer_backbone = ( Intra_layer_backbone >= 0.2).astype(int)
    Intra_layer_backbone_all = Intra_layer_backbone_all + Intra_layer_backbone
Intra_layer_backbone_all = Intra_layer_backbone_all / len( subjects )
fig = plt.figure()
plt.imshow(Intra_layer_backbone_all, cmap='seismic')
plt.colorbar()
fig.savefig('fig8.png')



# -------------------------------------------------- Plot a sample node embedding matrix
i_sub = 16
t = 2
fig, ax = plt.subplots(subjects[0].dFC.shape[0], 1, figsize=(5, 12))
ax = ax.ravel()
for layer in range( len(ax) ):
    plt.sca(ax[layer])
    plt.imshow(subjects[i_sub].layers[t][layer]['F'].real.transpose(), cmap='seismic')
    plt.ylabel('Embedding dimension')
    plt.xlabel('Nodes')
    plt.title(f'Layer #{layer + 1}')
    plt.setp(ax[layer].spines.values(), linewidth=2)
    ax[layer].set_aspect(0.7)
fig.suptitle('Node embeddings')
fig.tight_layout(h_pad=2)
fig.savefig('fig9.png')



# -------------------------------------------------- Plot a sample supra-adjacency matrix
t = 1
i_sub = 0
fig, ax = plt.subplots(1,1)
NoN_EEG_utils.plot_NoN_dFC( subjects[i_sub].NoN_dFC_latent_space[:,:, t], ax=ax, bin=False)
fig.savefig('fig10.png')



# -------------------------------------------------- Plot a sample comparison of real intra-layer FC and embedding-derived intra-layer connectivity
i_sub = 25
t = 10
fig = plt.figure(figsize=(5,5))
ax = plt.gca()

NoN_dFC_latent_space = subjects[i_sub].NoN_dFC_latent_space
FC = NoN_dFC_latent_space[:,:,t]
NoN_EEG_utils.plot_NoN_dFC(FC, L=5,ax=ax, bin=0)
fig.savefig('fig11.png')

FC_real = subjects[i_sub].dFC[:, :,:, t]
fig, ax = plt.subplots(1, FC_real.shape[0], figsize=(12,4))
ax = ax.ravel()
for freq in range(FC_real.shape[0]):
    ax[freq].imshow( FC_real[freq, :, :])
fig.savefig('fig12.png')


