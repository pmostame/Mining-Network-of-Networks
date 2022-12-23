import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import scipy as sp
from scipy.stats import zscore



# ______________________________________________________________________________________________________ SUBJECT FUNCTIONS
class Subject():
    def __init__(self, i_sub, file_dir):
        self.i_sub = i_sub
        self.file_dir = file_dir
        self.get_dFC()

    def get_dFC(self, T_samples=None):
        """
        Loads dynamic connectome data of the subject. 
        \n Input: 
        - T_samples: Desired length of data in samples
        \n Ouput:
        - None
        """
        f = h5py.File(self.file_dir)
        dFC_allfreqs = []
        for freq in range( len( f['B'] ) ):
            print('loading freq: ' + str(freq) + '\n')
            dFC_ref = f['B'][freq][0]
            dFC = np.array( f[dFC_ref], dtype='e')
            dFC = np.transpose(dFC, (1, 2, 0))
            if T_samples == None:
                T_samples = dFC.shape[-1]
            dFC_allfreqs.append( dFC[:, :, 0:T_samples:40] ) # make T_sampling = 4s from original 0.1s
        self.dFC = np.stack( dFC_allfreqs, axis=0) # stack FC patterns on a new axis and put that axis the first axis. (freq, elec, elec, time)

    def binarize_dFC(self, edge_count_Threshold=0.1):
        """
        Binarizes the dynamic connectome data of the subject. 
        \n Input: 
        - edge_count_Threshold: Percentage of edges passing the threshold
        \n Ouput:
        - dFC_bin: Binarized dynamic connectome data
        """
        dFC = self.dFC
        dFC_bin = np.empty( dFC.shape )
        N_edges = ( dFC.shape[1] * (dFC.shape[1] - 1) ) / 2
        K = np.floor( edge_count_Threshold * N_edges ).astype(int) # you should add N to it since diagonal is one!
        for t in range( dFC.shape[-1] ):
            for freq in range( dFC.shape[0] ):
                FC = dFC[freq, :, :, t]
                FC_threshold = np.sort( abs( FC.reshape(1, -1) ) )[0][-K]
                dFC_bin[freq, :, :, t] = (abs(FC) >= FC_threshold).astype(int)
        return dFC_bin

    def get_NoN(self, edge_count_Threshold=0.1, latent_dim=20, Null_Repeat = 200, threshold = 3):
        """
        Extracts supra-adjacency matrix of the subject across all timepoints. 
        \n Input: 
        - edge_count_Threshold: Percentage of edges passing the threshold
        - latent_dim: Length of embedding vector
        - Null_Repeat: number of surrogate data
        - threshold: threshold for statistical tests
        \n Ouput:
        - NoN_dFC_latent_space: supra-adjacency matrix
        - layers: all the variables extracted in MANE algorithm including F,M,A,D etc.
        - Intra_layer_backbone: the cross-layer dependency matrix used as D matrix in MANE.
        """   
        dFC_bin = self.binarize_dFC(edge_count_Threshold)
        new_N = dFC_bin.shape[0] * dFC_bin.shape[1] # freq * elec
        layers = [None] * dFC_bin.shape[-1]
        cross_layer_links = [None] * dFC_bin.shape[-1]
        NoN_dFC = np.empty( (new_N, new_N, dFC_bin.shape[-1]) )
        NoN_dFC_latent_space = np.empty( (new_N, new_N, dFC_bin.shape[-1]) )

        Intra_layer_backbone = np.mean(dFC_bin.mean(axis=-1), axis=0)
        Intra_layer_backbone = ( Intra_layer_backbone >= 0.2).astype(int)
        for t in tqdm( range( dFC_bin.shape[-1] ), desc='Time :'):
            FC = dFC_bin[:, :, :, t]            
            FC = FC + (np.tile(np.eye( FC.shape[1] ), (FC.shape[0], 1, 1))) # set diagonals to one to avoid inf in laplacian matrix
            layers[t] = MANE(FC, Intra_layer_backbone=Intra_layer_backbone, latent_dim=latent_dim)
            # cross_layer_links[t] = Extract_cross_layer_links(layers[t], Null_Repeat, threshold)
            # all_layers_FC = Extract_all_layers_FC(FC, cross_layer_links[t])
            # NoN_dFC[:, :, t] = all_layers_FC
            NoN_dFC_latent_space[:, :, t] = extract_latent_FC(layers[t])

        # return NoN_dFC, NoN_dFC_latent_space, cross_layer_links, layers
        self.NoN_dFC_latent_space = NoN_dFC_latent_space
        self.layers = layers
        self.Intra_layer_backbone = Intra_layer_backbone
        return NoN_dFC_latent_space, layers, Intra_layer_backbone








# ______________________________________________________________________________________________________ MANE FUNCTIONS
def MANE(G, Intra_layer_backbone=None, latent_dim=100, alpha=0.1, Max_iter = 200):
    """
    Performs MANE algorithm on a multi-layer network.
    \n Input: 
    - G: multi-layer network
    - Intra_layer_backbone: the cross-layer dependency matrix used as D matrix in MANE.
    - latent_dim: Length of embedding vector 
    - alpha: alpha variable in MANE
    \n Ouput:
    - layers: all the variables extracted in MANE algorithm including F,M,A,D etc.
    """
    # initialize the multiplex network
    g = G.shape[0]
    N = G.shape[1]
    layers = [{}] * g
    Dependencies_cross_layers = [ [None] * g ] * g
    if isinstance(Intra_layer_backbone, type(None)):
        Intra_layer_backbone = np.eye(N)
    # form within-layer variables
    for ind, layer in enumerate(layers):
        layer['name'] = ind
        layer['A'] = G[ind, :, :]
        layer['L'] = MANE_calculate_L( layer['A'] )
        layer['F'] = create_orthonormal_mat(N, latent_dim)
        layers[ind] = deepcopy(layer)
        # form cross-layer dependency matrix
        for otherind, _ in enumerate(layers):
            if ind != otherind:
                Dependencies_cross_layers[ind][otherind] = Intra_layer_backbone
                
    # iteratively solve for optimal F
    cost_prev = -1
    cost = 0
    counter = 0
    while (abs(cost - cost_prev) > 0.001) and (counter < Max_iter):
        counter = counter + 1
        layers = MANE_calculate_M(layers, Dependencies_cross_layers, alpha)
        layers = MANE_extract_F(layers, latent_dim)
        cost_prev = cost
        cost = MANE_cost_function(layers, Dependencies_cross_layers, alpha)
    return layers

def create_orthonormal_mat(m, n):
    H = np.random.rand(m, n)
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    return u @ vh

def MANE_cost_function(layers, D, alpha):
    cost = 0
    for ind, layer in enumerate(layers):
        cost = cost + np.trace( layer['F'].transpose() @ layer['L'] @ layer['F'])
        for otherind, otherlayer in enumerate(layers):
            K = layer['F'].transpose() @ D[ind][otherind] @ otherlayer['F']
            cost = cost - alpha * np.linalg.norm( D[ind][otherind] - (layer['F'] @ K @ otherlayer['F'].transpose()) )
    return cost

def MANE_calculate_L(A):
    H_inv = np.diag( 1 / np.sqrt( np.mean(A, axis=1) ) )
    return H_inv @ A @ H_inv

def MANE_calculate_M(layers, D, alpha):
    # calculate M of each layer by iterating over other layers
    for ind, layer in enumerate(layers):
        layer['M'] = np.empty( layer['L'].shape )
        temp = np.zeros( (layer['L'].shape) )
        for otherind, otherlayer in enumerate(layers):
            temp = temp + D[ind][otherind] @ otherlayer['F'] @ otherlayer['F'].transpose() @ D[ind][otherind].transpose()
        layer['M'] = layer['L'] + ( alpha * temp )
        # reassign the layer to output
        layers[ind] = deepcopy(layer)
    return layers
    
def MANE_extract_F(layers, latent_dim):
    for ind, layer in enumerate(layers):
        M = layer['M']
        w, v = np.linalg.eig(M)
        inds = w.argsort()[-latent_dim:][::-1]
        layer['F'] = v[:, inds]
        layers[ind] = deepcopy(layer)
    return layers

def Extract_cross_layer_links(layers, Null_Repeat = 50, threshold = 3):
    N = layers[0]['A'].shape[0]
    g = len(layers)
    cross_layer_latent_corr = [None] * g
    for ind, layer in enumerate(layers):
        cross_layer_latent_corr_temp = [{}] * g
        for otherind, otherlayer in enumerate(layers):
            cross_layer_temp = {}
            cross_layer_temp['real'] = np.corrcoef( np.vstack( (layer['F'], otherlayer['F']) ) )[:N, -N:].real
            if ind < otherind:
                # create null model
                cross_layer_temp['null'] = np.empty( cross_layer_temp['real'].shape + (Null_Repeat,) )
                for repeat in range(Null_Repeat):
                    Null_F = deepcopy(otherlayer['F'])
                    for row in range( otherlayer['F'].shape[0] ):
                        temp = Null_F[row, :]
                        np.random.shuffle( temp )
                        Null_F[row, :] = temp
                    cross_layer_temp['null'][:, :, repeat] = np.corrcoef( np.vstack( (layer['F'], Null_F) ) )[:N, -N:].real
                # assess significance
                temp = np.empty( cross_layer_temp['real'].shape )
                for node_layer in range( temp.shape[0] ):
                    for node_otherlayer in range( temp.shape[1] ):
                        temp[node_layer, node_otherlayer] = ( cross_layer_temp['real'][node_layer, node_otherlayer] - np.mean( cross_layer_temp['null'][node_layer, node_otherlayer, :] ) ) / ( np.std( cross_layer_temp['null'][node_layer, node_otherlayer, :] ) )
                temp = ( temp >= threshold ).astype(int)
                cross_layer_temp['sig'] = temp
            else:
                cross_layer_temp['null'] = None
                cross_layer_temp['sig'] = None
            cross_layer_latent_corr_temp[otherind] = deepcopy(cross_layer_temp)
        cross_layer_latent_corr[ind] = deepcopy(cross_layer_latent_corr_temp)
    return cross_layer_latent_corr

def Extract_all_layers_FC(FC_bin, cross_layer_links):
    N = len( cross_layer_links)
    all_layers_FC = [None] * N
    for ind in range(N):
        all_layers_FC_temp = [None] * N
        for otherind in range(N):
            if ind < otherind:
                all_layers_FC_temp[otherind] = cross_layer_links[ind][otherind]['sig'] * (1 - np.eye(FC_bin.shape[1]))
            elif ind == otherind:
                all_layers_FC_temp[otherind] = FC_bin[ind, :, :] * (1 - np.eye(FC_bin.shape[1]))
            else:
                all_layers_FC_temp[otherind] = cross_layer_links[otherind][ind]['sig'].transpose() * (1 - np.eye(FC_bin.shape[1]))
        all_layers_FC[ind] = np.concatenate(all_layers_FC_temp, axis=1)
    all_layers_FC = np.concatenate(all_layers_FC, axis=0)
    return all_layers_FC

def extract_latent_FC(layers):
    latent = np.vstack( [layers[ind]['F'] for ind in range( len(layers) )] )
    all_layers_FC_latent = np.corrcoef( latent ).real 
    all_layers_FC_latent = np.where(np.isnan(all_layers_FC_latent), 0, all_layers_FC_latent)
    all_layers_FC_latent = all_layers_FC_latent * (1 - np.eye(all_layers_FC_latent.shape[0]))
    return all_layers_FC_latent

def vectorize_FC(FC):
    inds = np.triu_indices(FC.shape[0], k=1)
    FC_vec = FC[inds]
    return FC_vec
    
def de_vectorize_FC(FC_vec, N):
    FC_vec.astype(float)
    FC = np.empty([N, N])
    for i in range(N):
        nans = np.zeros([1, i + 1])
        vals = FC_vec[0: N - i - 1].reshape(1, -1)
        FC_vec = np.delete(FC_vec, np.s_[:N - i - 1])
        FC[i, :] = np.hstack((nans, vals))
    # mirror FC
    FC = FC + FC.transpose()
    np.fill_diagonal(FC, np.nan)
    return FC









# ______________________________________________________________________________________________________ VISUALIZATION FUNCTIONS
def plot_NoN_dFC(NoN_dFC_t, L=5, ax=None, bin=1):
    """
    Plots the supra-adjacency matrix of a multi-layer network.
    \n Input: 
    - NoN_dFC_t: a supra-adjacency matrix with L layers
    - L: number of layers
    - ax: axis to be plotted on
    - bin: True for binary matrix, False for continuous matrix.
    \n Ouput:
    - None
    """
    if ax == None:
        plt.figure(figsize=(5,5))
        ax = plt.gca()
    N = int(NoN_dFC_t.shape[0] / L)
    # plot whole matrix
    plt.sca(ax)
    ax.set_aspect(1)
    # Define colors
    if bin:
        cmap = sns.color_palette("rocket", 2)
        vmin=0
        vmax=1
    else:
        cmap = sns.color_palette("seismic", as_cmap=True)
        vmin=-1
        vmax=1
    sns.heatmap(NoN_dFC_t, vmin=vmin, vmax=vmax, cmap=cmap, cbar_kws={'fraction':0.04, 'pad':0.02})
    # Set the colorbar labels
    if bin:
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25,0.75])
        colorbar.set_ticklabels(['0', '1'])
    # plot lines to seperate layers
    for i in range(1, L):
        for j in range(1, L):    
            ax.plot([i*N, i*N],[0, L*N], color='k', linewidth=0.2)
            ax.plot([0, L*N], [j*N, j*N], color='k', linewidth=0.2)
    # correct xticks and yticks
    tick_labels = np.tile(np.arange(0, N, 10).astype(int), L)
    tick_vals = np.concatenate([i*N + np.arange(0, N, 10).astype(int) for i in range(0, L)], axis=-1).astype(int)
    plt.xticks(ticks=tick_vals, labels=tick_labels, fontsize=6)
    plt.yticks(ticks=tick_vals, labels=tick_labels, fontsize=6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('k')

def performance_within_layer(Empirical, Recovered_NoN, Repeat=50, plot_flag=1):
    """
    Estimates Correaltion between empirical FC and recovered NoN within-layer connections. 
    \n Input: 
    - Empirical FC (freq, elec, elec, time)
    - Recovered FC NoN (freq*elec, freq*elec, time)
    \n Ouput:
    - Corrs: Z-scored values of correlation values with respect to null data, over layers and time (layer, time) 
    - Z-scored corresponding null distributions (layer, time, repeat)
    """
    freq_max = Empirical.shape[0]
    time_max = Recovered_NoN.shape[-1]
    Corrs = np.empty( (freq_max, time_max) )
    Corrs_null = np.empty( (freq_max, time_max, Repeat) )
    for t in range( time_max ):
        for freq in range( freq_max ):
            # extract empirical and recovered FCs
            FC_empirical = Empirical[freq,:,:, t]
            np.fill_diagonal(FC_empirical, 0)
            a = freq * FC_empirical.shape[0]
            b = a + FC_empirical.shape[0]
            FC_recovered = Recovered_NoN[a:b,a:b, t]
            # vectorize both FCs
            FC_empirical = np.tril(FC_empirical).flatten()
            FC_recovered = np.tril(FC_recovered).flatten()
            # Create null distribution
            for repeat in range(Repeat):
                inds = np.arange(FC_recovered.size)
                np.random.shuffle(inds)
                FC_recovered_null = np.take(FC_recovered, inds, axis=None)
                rho, pval = sp.stats.spearmanr( FC_recovered_null, FC_empirical, axis=None)
                Corrs_null[freq, t, repeat] = rho
            # estimate correlation
            rho, pval = sp.stats.spearmanr( FC_recovered, FC_empirical, axis=None)
            mu = Corrs_null[freq, t, :].mean()
            sigma = Corrs_null[freq, t, :].std()
            Corrs[freq, t] = (rho - mu) / sigma
    return Corrs, sp.stats.zscore(Corrs_null, axis=-1)

def plot_performance_within_layer(Corrs, Corrs_null):
    """
    Plot performance of intra-layer connection prediction of MANE
    \n Input: 
    - Corrs: Z-scored values of correlation values with respect to null data, over layers and time (layer, time) 
    - Z-scored corresponding null distributions (layer, time, repeat)
    \n Ouput:
    - None
    """
    fig = plt.figure(figsize=(8,15))
    t = np.arange(Corrs.shape[1])
    for freq in range( Corrs.shape[0] ):
        fig.add_subplot(Corrs.shape[0] + 1, 1, freq+1)
        _ = plt.plot(t , Corrs[freq, :], alpha=1, marker='s', markersize=3, color= plt.cm.tab10(freq), label='layer #' + str(freq + 1))
        for repeat in range( Corrs_null.shape[-1] ):
            _ = plt.scatter(t, Corrs_null[freq, :, repeat], alpha=0.1, s=1, color= plt.cm.tab10(freq))
        plt.ylim(-1, Corrs[freq, :].max() + 1)
        plt.xlabel('Time(s)')
        plt.ylabel('Z value')
        plt.xticks(ticks=np.arange(0, Corrs.shape[1], 10), labels=np.arange(0, Corrs.shape[1], 10))
        plt.title(f'Layer #{freq+1}')
    fig.add_subplot(Corrs.shape[0] + 1, 1, Corrs.shape[0] + 1)
    _ = plt.plot(t , Corrs.max(axis=0), alpha=1, marker='s', markersize=3, color= 'k', label='all layers')
    for repeat in range( Corrs_null.shape[-1] ):
        _ = plt.scatter(t, Corrs_null[:, :, repeat].max(axis=0), alpha=0.1, s=1, color= 'k')
    plt.ylim(-1, Corrs.max(axis=0).max() + 1)
    plt.xlabel('Time(s)')
    plt.ylabel('Z value')
    plt.xticks(ticks=np.arange(0, Corrs.shape[1], 10), labels=np.arange(0, Corrs.shape[1], 10))
    plt.title(f'All layers')
    fig.tight_layout(pad=0.5)  
    return fig