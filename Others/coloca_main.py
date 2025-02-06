
###########################################################################################
###########################################################################################
##### Main function for Colocalization Analysis
##### which considers to incorporate the precision of two emitters
from package_func import *
from filter_pre import *
from coloca_func import *
from support_func import *
from select_func import * 
from drift_func import * 

###########################################################################################
## One type of experimental data: Control Data (HALO_TM_SNAP)
## Experimental Data 20230502
## Maximum frame: 1482
## 1: Frame (0) 2: X/nm (1) 3: Y/nm (2) 4:PSF half width (3) 5: Number of Photons (4)
## 8: Precision/nm (5)
Filedate = '20230722'
Cell =  '20230109_Image1'
# path0 = '/Users/xcyan/Desktop/SSA/20230521_Images/Halo-TM-SNAP_100nm_beads/Image 4 Table.txt'
# path0 = '/Volumes/prigozhin_lab/Users/asrinivasan/Elyra_Imaging/20230109/DeltaS_2C_Halo_TM_SNAP/Image1_loc_table.txt'
path0 = 'Image1_loc_table.txt'
# path1 = '/Volumes/prigozhin_lab/Users/xyan/Elyra_Imaging/20230310/Halo549_B2AR_Gs_SNAP646/20230310_Image1/20230310_image1_emitter_increment.txt'
# path2 = '/Users/xcyan/Desktop/SSA2/Colocalization_Analysis/20230109_TM_image1_C1'
path2 = os.getcwd()
data = pd.read_csv(path0, sep='\t', encoding= 'unicode_escape').values[0:46819]                                                                                                                                                                                
## Filter Precision < 50 nm to use and Change the distance unit from nm to um
min_frame_use = 0
max_frame_use = 10000
frame_len = max(data[:,1])
R_data = data[(data[:,11]==1) & (data[:,1] >= min_frame_use) & (data[:,1]<= max_frame_use) & (data[:,6]<= 50)].astype(float)
G_data = data[(data[:,11]==2) & (data[:,1] >= min_frame_use) & (data[:,1]<= max_frame_use) & (data[:,6]<= 50)].astype(float)
R_pos = R_data[:,4:6]/1000
G_pos = G_data[:,4:6]/1000
## Frame Number 
R_frame = R_data[:,1]
R_precision = R_data[:,6]
R_PSF = R_data[:,10]
G_frame = G_data[:,1]
G_precision = G_data[:,6]
G_PSF = G_data[:,10]
pixel_size = 96.78
##### data[:,7]: Number of photon, data[:,10]: PSF half width, data[:,8]: Background Variance
mean_SNR = SNR_nonzero(data[:,7], data[:,10], data[:,8], pixel_size)

Filter_Operation = False
Drift_Operation = False
Selection_Number = 0

#################################################################################################################################
#### Proprocessing Part I: Filtering to keep good localizations to use
#### Filtering or load the existing filtered data table, PFO means: filterd by Precision + PSF + Overcounting
#### Can also filter one more based on intensity, see filter_func to modify
if Filter_Operation:
    R_pos_PFO, G_pos_PFO, R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, R_frame_PFO,  G_frame_PFO = \
        filter_all(data, pixel_size, R_data, G_data, R_prec_lower = 5, R_prec_upper = 40, G_prec_lower = 5, G_prec_upper = 40, \
                   R_psf_lower = 90, R_psf_upper = 250, G_psf_lower = 90, G_psf_upper = 250, overcount_thre = 0.075,\
                   path = path2 + f'/output_basic_info{Filedate}/', cell= Cell)
    save_files_filter(Cell, R_pos_PFO, G_pos_PFO, R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, R_frame_PFO, G_frame_PFO,  \
                    path = path2 + f'/output_basic_info{Filedate}/')
else: 
    ## As filtering takes some time for large data, load the already saved filtered data
    R_pos_PFO, G_pos_PFO, R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, R_frame_PFO, G_frame_PFO = \
        load_files_filter(Cell, path = path2 + f'/output_basic_info{Filedate}/')

############################################################################################################
### Drift correction, read the emitter tracjectories from files
## use the first 3000 frames of the images, in unit nm
if Drift_Operation:
    emitter_incre_pos = np.loadtxt(path1)[:,1:3]*pixel_size/1000
    # total_frame = int(max(data[:,1]))
    # frame_array = np.cumsum(np.ones((total_frame, 1)))
    frame_array = np.loadtxt(path1)[:,0]
    emitter_incre = np.vstack([frame_array, emitter_incre_pos[:,0], emitter_incre_pos[:,1]])
    # Choose the start frame and end frame in drift correction 
    frame_start = min(frame_array)
    frame_end = max(frame_array)
    emitter1_incre = emitter_incre[:, (emitter_incre[:][0] >= frame_start) & (emitter_incre[:][0] <= frame_end)]
    emitter1_incre = np.transpose(emitter1_incre)
    R_pos_driftcorr, G_pos_driftcorr = drift_correction_new2(emitter1_incre, R_pos_PFO, G_pos_PFO, R_frame_PFO, G_frame_PFO, plotting = True)
    ### The other variables precision, psf and frame don't change 
    R_prec_driftcorr, G_prec_driftcorr, R_psf_driftcorr, G_psf_driftcorr, R_frame_driftcorr, G_frame_driftcorr = \
        R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, R_frame_PFO, G_frame_PFO  
    save_files_drift(Cell, R_pos_driftcorr, G_pos_driftcorr, R_prec_driftcorr, G_prec_driftcorr, R_psf_driftcorr, 
            G_psf_driftcorr, R_frame_driftcorr, G_frame_driftcorr, path = path2 +f'/output_basic_info{Filedate}/')
else:
    #######Originally use the function load_files_drift, but if we don't do drift correction, then load_files_filter
    R_pos_driftcorr, G_pos_driftcorr, R_prec_driftcorr, G_prec_driftcorr, R_psf_driftcorr, G_psf_driftcorr, R_frame_driftcorr, G_frame_driftcorr = \
        load_files_filter(Cell, path = path2 + f'/output_basic_info{Filedate}/')

#################################################################################################################
####### R_select, G_select is the selected position
# R_index_analyze, G_index_analyze, area = area_selection(R_pos_driftcorr, G_pos_driftcorr, Cell, filedate = Filedate, 
#            path_save = path2 + f'/output_basic_info{Filedate}', selection_number = Selection_Number)
R_index_analyze = np.loadtxt(path2 + f"/output_basic_info{Filedate}/Area Selection Number{Selection_Number}/R_select_idx_cell20230109_Image1_for_Selection{Selection_Number}.txt")
G_index_analyze = np.loadtxt(path2 + f"/output_basic_info{Filedate}/Area Selection Number{Selection_Number}/G_select_idx_cell20230109_Image1_for_Selection{Selection_Number}.txt")
R_pos_analyze = R_pos_driftcorr[R_index_analyze.astype(int)]
G_pos_analyze = G_pos_driftcorr[G_index_analyze.astype(int)]
R_prec_analyze = R_prec_driftcorr[R_index_analyze.astype(int)]
G_prec_analyze = G_prec_driftcorr[G_index_analyze.astype(int)]
R_frame_analyze = R_frame_driftcorr[R_index_analyze.astype(int)]
G_frame_analyze = G_frame_driftcorr[G_index_analyze.astype(int)]
R_psf_analyze = R_psf_driftcorr[R_index_analyze.astype(int)]
G_psf_analyze = G_psf_driftcorr[G_index_analyze.astype(int)]

save_files_analyze(R_pos_analyze, G_pos_analyze, R_prec_analyze, G_prec_analyze, R_psf_analyze, G_psf_analyze, R_frame_analyze, G_frame_analyze, \
        path = path2 + f'/output_basic_info{Filedate}/', cell= Cell, selection_number = Selection_Number)

##########################################################################################################################################
########## Second Part: Cololization Analysis
########## Make sure to have the correct input

file_path = f"output_basic_info{Filedate}/Area Selection Number{Selection_Number}/info_cell20230109_Image1_for_Selection{Selection_Number}.txt"
# Open the file in read mode
with open(file_path, "r") as file:
    # Read each line in the file
    for line in file:
        # Check if the line contains the keyword "area"
        if "area" in line.lower():
            # Extract the area number from the line (assuming it is a number)
            area_number = float(line.split(":")[1].strip())
            break  # Exit the loop since we found the area information

exp_num_pair, MC_num_hist, pair_num_est = \
    run_exp_iterative_MC(R_pos_analyze, G_pos_analyze, R_prec_analyze, G_prec_analyze, \
            R_frame_analyze , G_frame_analyze, area = area_number, number_out_iter = 15, number_in_iter = 5, \
            frame_len = frame_len, path = path2, cell = Cell, filedate = Filedate, \
            d_true_thre = 20/1000, dis_tree_thre = 150/1000, num_MC_points = int(1e5))

