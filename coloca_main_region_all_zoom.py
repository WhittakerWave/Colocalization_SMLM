
#################################################################################################################################
#################################################################################################################################
##### Main function for Colocalization Analysis for HALO-TM-SNAP data
##### which considers to incorporate the precision of two emitters
from package_func import *
from filter_pre import *
from coloca_func import *
from support_func import *
from select_func import * 
from drift_func import * 
from coloca_region import *
from region_select import *
from delete_cluster import *

#################################################################################################################################
## One type of experimental data: Control Data (HALO_TM_SNAP)
## Experimental Data 20230502
## Maximum frame: 
## 1: Frame (0) 2: X/nm (1) 3: Y/nm (2) 4:PSF half width (3) 5: Number of Photons (4)
## 8: Precision/nm (5)
Filedate = '20240912_R2'
Cell =  '20231102_Image3'
# path0 = '/Users/xcyan/Desktop/SSA/20230521_Images/Halo-TM-SNAP_100nm_beads/Image 4 Table.txt'
# path0 = '/Volumes/prigozhin_lab/Users/asrinivasan/Elyra_Imaging/20230109/DeltaS_2C_Halo_TM_SNAP/Image2_loc_table.txt'
path0 =  '20231102_B2CAAX_Image 3_dc_filtered_grp.txt'
# path1 = '/Volumes/prigozhin_lab/Users/xyan/Elyra_Imaging/20230310/Halo549_B2AR_Gs_SNAP646/20230310_Image1/20230310_image1_emitter_increment.txt'
# path2 = '/Users/xcyan/Desktop/SSA2/Colocalization_Analysis/20230109_TM_image2_C1'
path2 = os.getcwd()
data = pd.read_csv(path0, sep='\t', encoding= 'unicode_escape').values[0:3273]                                                                                                                                                                                                       
## Filter Precision < 50 nm to use and Change the distance unit from nm to um
min_frame_use = 0
max_frame_use = 10000
frame_len = max(data[:,1])
R_data = data[(data[:,11]==1) & (data[:,1] >= min_frame_use) & (data[:,1]<= max_frame_use) & (data[:,6]<= 50)].astype(float)
G_data = data[(data[:,11]==2) & (data[:,1] >= min_frame_use) & (data[:,1]<= max_frame_use) & (data[:,6]<= 50)].astype(float)
R_pos = R_data[:,4:6]/1000
G_pos = G_data[:,4:6]/1000
## Frame Number; Min precision: 7nm/7nm; Max precision: 35nm/45nm
## Min PSF: 90nm/83.6nm; Max PSF: 180nm/180nm
R_frame = R_data[:,1]
R_precision = R_data[:,6]
R_PSF = R_data[:,10]
G_frame = G_data[:,1]
G_precision = G_data[:,6]
G_PSF = G_data[:,10]
pixel_size = 96.78
##### data[:,7]: Number of photon, data[:,10]: PSF half width, data[:,8]: Background Variance
mean_SNR = SNR_nonzero(data[:,7], data[:,10], data[:,8], pixel_size)

Filter_Operation = True
Drift_Operation = False
Selection_Number = 0
Delete_Cluster = True
Subregion_Operation = True
num_rows = 10
num_cols = 10
matrix_size = num_rows*num_cols 

#################################################################################################################################
#### Proprocessing Part I: Filtering to keep good localizations to use
#### Filtering or load the existing filtered data table, PFO means: filterd by Precision + PSF + Overcounting
#### Can also filter one more based on intensity, see filter_func to modify
if Filter_Operation:
    ### I have turned off overcounting filtering 
    R_pos_PFO, G_pos_PFO, R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, R_frame_PFO, G_frame_PFO = \
        filter_all(data, pixel_size, R_data, G_data, R_prec_lower = 5, R_prec_upper = 40, \
            G_prec_lower = 5, G_prec_upper = 40, R_psf_lower = 80, R_psf_upper = 220, \
            G_psf_lower = 80, G_psf_upper = 220, overcount_thre = 0, \
            path = path2 + f'/output_basic_info{Filedate}/', cell= Cell)
    save_files_filter(Cell, R_pos_PFO, G_pos_PFO, R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, \
            R_frame_PFO, G_frame_PFO,  \
            path = path2 + f'/output_basic_info{Filedate}/')
else: 
    ##As filtering takes some time for large data, load the already saved filtered data
    R_pos_PFO, G_pos_PFO, R_prec_PFO, G_prec_PFO, R_psf_PFO, G_psf_PFO, R_frame_PFO, G_frame_PFO = \
        load_files_filter(Cell, path = path2 + f'/output_basic_info{Filedate}/')

#################################################################################################################################
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
    #######Originally use the function load_files_drift, 
    ####### but if we don't do drift correction, then load_files_filter, i.e, load files from filterings
    R_pos_driftcorr, G_pos_driftcorr, R_prec_driftcorr, G_prec_driftcorr, R_psf_driftcorr, G_psf_driftcorr, R_frame_driftcorr, G_frame_driftcorr = \
        load_files_filter(Cell, path = path2 + f'/output_basic_info{Filedate}/')
    

'''
# create a SelectPoints object and show the plot
width = 3
height = 3
rs = RegionSelector(R_pos_driftcorr, G_pos_driftcorr, cellname = Cell, width=width, height=height, path_select = path2 + f'/output_basic_info{Filedate}/Subregion_Selection/')
plt.show()
R_select_index = rs.select_indices_R 
G_select_index = rs.select_indices_G
save_files_subregion(Cell, R_select_index, G_select_index,  R_pos_driftcorr, G_pos_driftcorr, R_prec_driftcorr, G_prec_driftcorr, \
        R_psf_driftcorr, G_psf_driftcorr, R_frame_driftcorr, G_frame_driftcorr, path = path2 + f'/output_basic_info{Filedate}/Subregion_Selection/')
'''

# Combine R_frame, R_pos, R_prec, R_psf into R_data
R_data_combine = np.column_stack((R_frame_driftcorr, R_pos_driftcorr, R_prec_driftcorr, R_psf_driftcorr))
# Combine G_frame, G_pos, G_prec, G_psf into G_data
G_data_combine  = np.column_stack((G_frame_driftcorr, G_pos_driftcorr, G_prec_driftcorr, G_psf_driftcorr))

if Delete_Cluster == True: 
    R_pos_delete, G_pos_delete, R_prec_delete, G_prec_delete, R_frame_delete, \
        G_frame_delete, R_psf_delete, G_psf_delete, R_conv_coord, G_conv_coord = \
        delete_cluster(R_data_combine, G_data_combine, eps = 75/1000, min_samples = 10, \
            path = f'output_basic_info{Filedate}/')
else:
    R_index_delete = np.array([int(line.strip()) for line in open(f'output_basic_info{Filedate}/R_delete_indice_cluster.txt', 'r')])
    G_index_delete = np.array([int(line.strip()) for line in open(f'output_basic_info{Filedate}/G_delete_indice_cluster.txt', 'r')])
    R_pos_delete, G_pos_delete, R_prec_delete, G_prec_delete, \
        R_frame_delete, G_frame_delete, R_psf_delete, G_psf_delete =\
            delete_cluster_load(R_data_combine, G_data_combine, R_index_delete, G_index_delete)

#################################################################################################################################
########## Second Part: Cololization Analysis
########## Make sure to have the correct input

if Subregion_Operation ==True:
    R_select, G_select, area_select, R_points_region, G_points_region, intersect_area, \
        R_points_intersect_index, G_points_intersect_index = \
        area_select_subregions(R_pos_delete, G_pos_delete, indicator = "exp", cell = Cell, \
            filedate = Filedate, path_save = os.path.join(path2, f'output_basic_info{Filedate}/Whole_Cell_Region'), \
           num_rows = num_rows, num_cols = num_cols, vertices = None)
else:
    ##### Loaded the saved selected regions fikes
    # path3 = os.path.join(path2, f'output_basic_info{Filedate}/Whole_Cell_Region/')
    # R_points_intersect_index = np.load(path3 + f'R_second_select_idx_region_cell{Cell}.npy', allow_pickle=True)
    # G_points_intersect_index = np.load(path3 + f'G_second_select_idx_region_cell{Cell}.npy', allow_pickle=True)
    # intersect_area = np.load(path3+ f"intersection_area.npy")
    vertices_prev = np.loadtxt(path2 + f'/output_basic_info{Filedate}/Whole_Cell_Region/Clicked_Vertices_for_Selection{Cell}.txt')
    R_select, G_select, area_select, R_points_region, G_points_region, intersect_area, \
        R_points_intersect_index, G_points_intersect_index = \
        area_select_subregions(R_pos_delete, G_pos_delete, indicator = "prev", cell = Cell, \
            filedate = Filedate, path_save = os.path.join(path2, f'output_basic_info{Filedate}/Whole_Cell_Region'), \
           num_rows = num_rows, num_cols = num_cols, vertices = vertices_prev)

####  local_density_analysis(R_select_filter, G_select_filter, R_points_region, G_points_region, intersect_area, area_filter, cell = Cell, filedate = Filedate, path_save = path1)
pairs_region_initial = np.zeros([num_rows, num_cols]) 
pairs_region_true = np.zeros([num_rows, num_cols]) 
num_R_regions = np.zeros([num_rows, num_cols]) 
num_G_regions = np.zeros([num_rows, num_cols]) 
density_R_regions = np.zeros([num_rows, num_cols]) 
density_G_regions = np.zeros([num_rows, num_cols]) 

try:
    # Attempt to create the new directory
    os.mkdir(path2 + f'/coloca_results{Filedate}/Frame_Diff_Region')
except FileExistsError:
    # Directory already exists, do nothing
    pass


for i in range(num_rows):
    for j in range(num_cols):
        if intersect_area[i,j] > 0:
            R_index = R_points_intersect_index[i, j]
            G_index = G_points_intersect_index[i, j]
            if R_index is None:
                num_R_regions[i,j] = 0
            else:
                num_R_regions[i,j] = len(R_index)
            if G_index is None:
                num_G_regions[i,j] = 0
            else:
                num_G_regions[i,j] = len(G_index)
            density_R_regions[i,j] = num_R_regions[i,j] / intersect_area[i,j]
            density_G_regions[i,j] = num_G_regions[i,j] / intersect_area[i,j]

with open(os.path.join(path2, f'coloca_results{Filedate}/Area_subregions.txt'), 'w') as f:
    for row in intersect_area:
        f.write(' '.join(str(elem) for elem in row))
        f.write('\n')
######### density_R_regions
with open(os.path.join(path2, f'coloca_results{Filedate}/Density_R_subregions.txt'), 'w') as f:
    for row in density_R_regions:
        f.write(' '.join(str(elem) for elem in row))
        f.write('\n')
######## density_G_regions
with open(os.path.join(path2, f'coloca_results{Filedate}/Density_G_subregions.txt'), 'w') as f:
    for row in density_G_regions:
        f.write(' '.join(str(elem) for elem in row))
        f.write('\n')
######### number_R_regions
with open(os.path.join(path2, f'coloca_results{Filedate}/Number_R_subregions.txt'), 'w') as f:
    for row in num_R_regions:
        f.write(' '.join(str(elem) for elem in row))
        f.write('\n')
######### number_G_regions
with open(os.path.join(path2, f'coloca_results{Filedate}/Number_G_subregions.txt'), 'w') as f:
    for row in num_G_regions:
        f.write(' '.join(str(elem) for elem in row))
        f.write('\n')

#################################################################################################################################
# Create an empty 2D list (list of lists) to indicate which method to use
matrix_indicator = []
# Define the size of your matrix, for example, 3x3
# Initialize the matrix with None values
### Matrix values: 0 (non processed); 1 graph; 2: non-graph 
for i in range(matrix_size):
    row = [0] * matrix_size
    matrix_indicator.append(row)

for i in range(num_rows):
    for j in range(num_cols):
        if intersect_area[i,j] > 0:
            R_index = R_points_intersect_index[i, j]
            G_index = G_points_intersect_index[i, j]
            if R_index is None:
                num_R_regions[i,j] = 0
            else:
                num_R_regions[i,j] = len(R_index)
            if G_index is None:
                num_G_regions[i,j] = 0
            else:
                num_G_regions[i,j] = len(G_index)
            density_R_regions[i,j] = num_R_regions[i,j] / intersect_area[i,j]
            density_G_regions[i,j] = num_G_regions[i,j] / intersect_area[i,j]

        ############## if there are R and G points in the regions and the maximum of the density of R and G <=80: 
        if R_points_intersect_index[i, j] is not None and G_points_intersect_index[i, j] is not None \
            and intersect_area[i,j]>0 and max(density_R_regions[i,j], density_G_regions[i,j]) <= 80: 
            R_index = R_points_intersect_index[i, j]
            G_index = G_points_intersect_index[i, j]
            R_pos_analyze = R_pos_delete[R_index]
            G_pos_analyze = G_pos_delete[G_index]
            R_prec_analyze = R_prec_delete[R_index]/1000
            G_prec_analyze = G_prec_delete[G_index]/1000
            R_frame_analyze = R_frame_delete[R_index]
            G_frame_analyze = G_frame_delete[G_index]
            exp_num_pair, MC_num_hist, pair_num_est = \
                run_exp_iterative_MC(R_pos_analyze, G_pos_analyze, R_prec_analyze, G_prec_analyze, \
                    R_frame_analyze , G_frame_analyze, area = intersect_area[i,j], number_out_iter = 25, number_in_iter = 10, \
                    frame_len = frame_len, path = path2, cell = f"Sub_Selection_Region{(i,j)}", filedate = Filedate, \
                    d_true_thre = 20/1000, dis_tree_thre_factor = 4, num_MC_points = int(1e5))
            pairs_region_initial[i,j] = exp_num_pair
            pairs_region_true[i,j] = pair_num_est[-1]
            matrix_indicator[i][j] = 1
            print(exp_num_pair, MC_num_hist, pair_num_est)
        ############## if there are R and G points in the regions and the maximum of the density of R and G >=80: Using approaximation ways
        ############## Find closest indexes        
        elif R_points_intersect_index[i, j] is not None and G_points_intersect_index[i, j] is not None \
            and intersect_area[i,j]>0 and max(density_R_regions[i,j], density_G_regions[i,j]) >= 80: 
            R_index = R_points_intersect_index[i, j]
            G_index = G_points_intersect_index[i, j]
            R_pos_analyze = R_pos_delete[R_index]
            G_pos_analyze = G_pos_delete[G_index]
            R_prec_analyze = R_prec_delete[R_index]/1000
            G_prec_analyze = G_prec_delete[G_index]/1000
            R_frame_analyze = R_frame_delete[R_index]
            G_frame_analyze = G_frame_delete[G_index]
            exp_num_pair, MC_num_hist, pair_num_est = \
                run_exp_iterative_MC_fixed_thre(R_pos_analyze, G_pos_analyze, R_prec_analyze, G_prec_analyze, \
                    R_frame_analyze , G_frame_analyze, threshold = 100/1000, area = intersect_area[i,j], \
                    number_out_iter = 25, number_in_iter = 10, \
                    frame_len = frame_len, path = path2, cell = f"Sub_Selection_Region{(i,j)}", \
                    filedate = Filedate, num_MC_points = int(1e5))
            pairs_region_initial[i,j] = exp_num_pair
            pairs_region_true[i,j] = pair_num_est[-1]
            matrix_indicator[i][j] = 2
            print(exp_num_pair, MC_num_hist, pair_num_est)

    with open(os.path.join(path2, f'coloca_results{Filedate}/Num_Pairs_Initial_subregions.txt'), 'w') as f:
        for row in pairs_region_initial:
            f.write(' '.join(str(elem) for elem in row))
            f.write('\n')
    with open(os.path.join(path2, f'coloca_results{Filedate}/Num_Pairs_True_subregions.txt'), 'w') as f:
        for row in pairs_region_true:
            f.write(' '.join(str(elem) for elem in row))
            f.write('\n')
    # Now, you can write the matrix to a file
    with open(os.path.join(path2, f'coloca_results{Filedate}/Method.txt'), 'w') as f:
        for row in matrix_indicator:
            # Convert the row to a string and write it to the file
            row_str = ' '.join(map(str, row))
            f.write(row_str + '\n')
