
###########################################################################################
##### function to filter Precison, PSF and overcounts
##### For any errors or questions, contact xingchiyan22@gmail.com

###########################################################################################
## Preprocessing Files
from package_func import *
from support_func import *
########################################################################################################
## Fucntion used to filter overcount or over-localizations
## Here overcount refers to one point appears in many consective frames (with some shifts)
## We delete one of the any two localizations in two consective frames whose pairwise distance are within the thresholds.
def filter_overcount(Pos, Frame, threshold):   
    ## KDTree on positions  
    tree = KDTree(Pos)
    pairs = tree.query_radius(Pos, threshold)
    index_delete = []
    ## The outer loop iterates over the pairs array in reverse order
    ## When an element is added to the index_delete array
    ## it is also removed from the pairs array using a list comprehension 
    for i in range(len(pairs)-1, -1, -1):
        if i in index_delete:
            continue
        ## Take Frame[i] for the i-th row
        seen = {}
        seen[i] = Frame[i]
        ## For i-th row, find all Frames[pairs[i][]] without replication with Frame[i]
        for j in range(len(pairs[i])):
            if pairs[i][j] != i:
                seen[pairs[i][j]] = Frame[pairs[i][j]]
        ## Exchange value to key
        value_to_key = {v: k for k, v in seen.items()}
        ## Sorted the values by order
        sorted_values = sorted([seen[k] for k in seen.keys()])
        current_group = [sorted_values[0]]
        for i in range(1, len(sorted_values)):
            ## If same or difference = 1, delete the previous index
            if sorted_values[i] - sorted_values[i-1] == 1 or sorted_values[i] - sorted_values[i-1] == 0:
                index_delete.append(value_to_key[sorted_values[i-1]])
            else:
                current_group.append(sorted_values[i])
    ## Filtered index = Total index - index_delete
    filtered_index = [i for i in range(len(Pos)) if i not in index_delete]
    return  filtered_index

###############################################################################################################
##### Function used to filter the intensity to avoid suspicious features 
def filter_intensity(data):
    # Step 1: Calculate the median of the 7th column
    median = np.median(data[:,7])
    # Step 2: Calculate the absolute deviation from the median for each value in the 7th column
    abs_dev = np.abs(data[:,7] - median)
    # Step 3: Calculate the median absolute deviation (MAD)
    mad = np.median(abs_dev)
    # Step 4: Choose a threshold value
    threshold =  20 * mad
    # Step 5: Filter the R_data array based on the threshold value
    filtered_data = data[np.abs(data[:, 7] - median) <= threshold]
    return filtered_data

##############################################################################################################
##### Main function to do filtering, three major filterings 

def filter_all(data, pixel_size, R_data, G_data, R_prec_lower, R_prec_upper, G_prec_lower, G_prec_upper, R_psf_lower, R_psf_upper, G_psf_lower, G_psf_upper, overcount_thre, path, cell):
    mean_SNR = SNR_nonzero(data[:,7], data[:,10], data[:,8], pixel_size)
    ### Filter data based on the intensity
    # R_data = filter_intensity(R_data)
    # G_data = filter_intensity(G_data)
    ## Filter Precision < 35/40 nm to use, here we choose 40 nm as threshold
    R_data_filter_P = R_data[(R_data[:,6] <= R_prec_upper) & (R_data[:,6] >= R_prec_lower)]
    G_data_filter_P = G_data[(G_data[:,6] <= G_prec_upper) & (G_data[:,6] >= G_prec_lower)]
    R_Frame = R_data_filter_P[:,1]
    R_precision_P = R_data_filter_P[:,6]
    R_PSF = R_data_filter_P[:,10]
    G_Frame = G_data_filter_P[:,1]
    G_precision_P = G_data_filter_P[:,6]
    G_PSF = G_data_filter_P[:,10]
    ## Filter out the PSF with 100nm (can be calculated from NA, wavelength, etc)
    min_frame = 0
    #max_frame = max(np.max(R_data_filter_PO[:,0]), np.max(G_data_filter_PO[:,0]))
    max_frame = max(np.max(R_data_filter_P[:,1]), np.max(G_data_filter_P[:,1]))
    ## Filter the PSF to be greater than 100
    Lambda_R = 561
    Lambda_G = 646
    NA = 1.4
    # psf_thre_R = Lambda_R/(4*NA)
    # psf_thre_G = Lambda_G/(4*NA)
    # psf_thre_G = Lambda_G/(2*NA)/(2*np.sqrt(2*np.log(2)))
    ### Filter frames for visualization and filter PSF
    idx_R_filter_psf = (R_data_filter_P[:,1] <= max_frame) & (R_data_filter_P[:,1] >= min_frame) \
        & (R_data_filter_P[:,10] >= R_psf_lower) & (R_data_filter_P[:,10] <= R_psf_upper) 
    idx_G_filter_psf = (G_data_filter_P[:,1] <= max_frame) & (G_data_filter_P[:,1] >= min_frame) \
        & (G_data_filter_P[:,10] >= G_psf_lower) & (G_data_filter_P[:,10] <= G_psf_upper) 
    R_data_filter_PF = R_data_filter_P[idx_R_filter_psf]
    R_precision_PF = R_precision_P[idx_R_filter_psf]
    G_data_filter_PF = G_data_filter_P[idx_G_filter_psf]
    G_precision_PF = G_precision_P[idx_G_filter_psf]
    ## Filter the overcounting in space with threshold 15nm/30nm, denote this filter as "O" 
    # idx_R_filter_O = filter_overcount(R_data_filter_PF[:,4:6]/1000, R_Frame, overcount_thre)
    # idx_G_filter_O  =  filter_overcount(G_data_filter_PF[:,4:6]/1000, G_Frame, overcount_thre)
    ## Final position after filtering 1) Precision (P)  2) PSF halfwidth (F) 3) Overlocalization (O)
    ## Not filtering overcouting for fast 
    R_data_PFO = R_data_filter_PF
    G_data_PFO = G_data_filter_PF
    # R_data_PFO = R_data_filter_PF[idx_R_filter_O]
    # G_data_PFO = G_data_filter_PF[idx_G_filter_O]
    R_pos_PFO = R_data_PFO[:,4:6]/1000
    G_pos_PFO = G_data_PFO[:,4:6]/1000
    R_prec_PFO = R_data_PFO[:,6]
    G_prec_PFO = G_data_PFO[:,6]
    R_psf_PFO = R_data_PFO[:,10]
    G_psf_PFO = G_data_PFO[:,10]
    R_frame_PFO = R_data_PFO[:,1]
    G_frame_PFO = G_data_PFO[:,1]
    R_len_before = len(R_data)
    G_len_before = len(G_data)
    R_len_after = len(R_data_PFO)
    G_len_after = len(G_data_PFO)

    ###################################### 
    info = [mean_SNR, R_prec_lower, R_prec_upper, G_prec_lower, G_prec_upper, R_psf_lower, R_psf_upper, G_psf_lower, G_psf_upper, overcount_thre, \
        R_len_before, G_len_before, R_len_after, G_len_after ]
    info_names = ['Mean SNR' ,'R precision lower/nm', 'R precision upper/nm', 'G precision lower/nm', 'G precision upper/nm', \
        'R PSF Lower/nm', 'R PSF Upper/nm', 'G PSF Lower/nm', 'G PSF Upper/nm', 'Overcount Threshold/um', \
          'R number before filtering',   'G number before filtering', \
          'R number after filtering',   'G number after filtering' ]
    with open(path + f'filtering_info_cell{cell}.txt', 'w') as f:
           for name, value in zip(info_names, info):
               print(f'{name}: {value}', file=f)
    return R_pos_PFO, G_pos_PFO, R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, R_frame_PFO,  G_frame_PFO

########################################################################################################
#### function to save the filtered files 
def save_files_filter(Cell, R_pos_PFO, G_pos_PFO, R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, R_frame_PFO,  G_frame_PFO, path):
    R_data_filtered  = np.column_stack((R_frame_PFO, R_pos_PFO[:, 0], R_pos_PFO[:, 1], R_prec_PFO, R_psf_PFO))
    G_data_filtered  = np.column_stack((G_frame_PFO, G_pos_PFO[:, 0], G_pos_PFO[:, 1], G_prec_PFO, G_psf_PFO))
    column_names = ['Frame', 'pos_x/um', 'pos_y/um', 'precision/nm', 'PSF/nm']
    R_data_filtered = np.vstack((column_names, R_data_filtered))
    G_data_filtered = np.vstack((column_names, G_data_filtered))
    np.savetxt(path + f'R_data_PFO{Cell}.txt', R_data_filtered, fmt='%s', delimiter='\t')
    np.savetxt(path + f'G_data_PFO{Cell}.txt', G_data_filtered, fmt='%s', delimiter='\t')

################################################################################################################
#### function to load the filtered files 

def load_files_filter(Cell, path):
    ### Load the files from preprocessing Filtering PFO 
    R_data_PFO = np.loadtxt(path +f'R_data_PFO{Cell}.txt', skiprows=1)
    G_data_PFO = np.loadtxt(path +f'G_data_PFO{Cell}.txt', skiprows=1)
    # Extract the columns from the data array
    R_frame_PFO = R_data_PFO[:, 0]
    R_pos_PFO = R_data_PFO[:, 1:3]
    R_prec_PFO = R_data_PFO[:, 3]
    R_psf_PFO = R_data_PFO[:, 4]
    G_frame_PFO = G_data_PFO[:, 0]
    G_pos_PFO = G_data_PFO[:, 1:3]
    G_prec_PFO = G_data_PFO[:, 3]
    G_psf_PFO = G_data_PFO[:, 4]
    return R_pos_PFO, G_pos_PFO, R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, R_frame_PFO, G_frame_PFO


