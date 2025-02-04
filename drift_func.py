
########################################################################################################
##### Drift Correction
from package_func import *

##### Use moving window for drift correction to average out the noise
def moving_window(data, window_size):
    # Calculate the moving average with a window size of 10
    weights = np.repeat(1.0, window_size) / window_size
    moving_average = np.apply_along_axis(lambda x: np.convolve(x, weights, 'same'), axis = 0, arr = data)
    return moving_average

###################################################################################################
##### Function to plot the drifts 

def drift_correction_plot(time, final_diff, indicator):
    ## Drift Correction Plot
    fig, ax = plt.subplots()
    ax.plot(time, final_diff)
    #ax.plot(time, np.cumsum(final_diff))
    ax.set_xlabel(r"Frame")
    ax.set_ylabel(r"Corrected Drift/um")
    ax.set_title(f"{indicator}-direction Total Corrected Drift", fontsize = 20)
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large', fontproperties=None, LW=2.3) 

###################################################################################################
### Functions to do drift correction

def drift_correct_new(position, frame, final_diff_x, final_diff_y, frame_bead):
    ## Drift Correction code
    pos = np.copy(position)
    #### extracts a portion of the frame_bead list, excluding the last element
    for item in set(frame_bead[:-1]):
        ## Find the index of the current frame in the position array
        ## start index
        start_index = min(frame_bead)
        ## index is the index of the frame array where the frame number is equal to the item
        ## this index corresponds to positions 
        idx = np.where(frame == item)[0].tolist()
        ## Question ? should we minus start index
        ## final_diff_x and final_diff_y corresponding to the frame_bead starting index
        ## for example, frame_bead starts from 501
        if len(idx) == 0: 
            continue
        elif len(idx) != 0:
            pos[idx,0] += final_diff_x[int(item - start_index)]
            pos[idx,1] += final_diff_y[int(item - start_index)]
    return pos


def drift_correction_new2(emitter, R_pos_PFO, G_pos_PFO, R_frame_PFO, G_frame_PFO, plotting):
    ##### The drift increment file that we save are increments of the positions, so there is a minus sign
    final_diff_x = -emitter[:,1]
    final_diff_y = -emitter[:,2]
    if plotting == True:
        # drift_plot_time = np.arange(len(final_diff_x))
        drift_plot_time = emitter[:,0]
        drift_correction_plot(drift_plot_time, final_diff_x, indicator = 'x')
        drift_correction_plot(drift_plot_time, final_diff_y, indicator = 'y')
    ##### Input: position, frame, adding final_diff_x and final_diff_y compared with starting index from emitter[:,0]
    R_pos_correct  = drift_correct_new(R_pos_PFO, R_frame_PFO, final_diff_x , final_diff_y, emitter[:,0])
    G_pos_correct  = drift_correct_new(G_pos_PFO, G_frame_PFO, final_diff_x , final_diff_y, emitter[:,0])
    return R_pos_correct, G_pos_correct



########################################################################################################
#### Function to save the drift files 

def save_files_drift(Cell, R_pos_driftcorr, G_pos_driftcorr, R_prec, G_prec, R_psf, G_psf, R_frame,  G_frame, path):
    R_data_filtered  = np.column_stack((R_frame, R_pos_driftcorr[:, 0], R_pos_driftcorr[:, 1], R_prec, R_psf))
    G_data_filtered  = np.column_stack((G_frame, G_pos_driftcorr[:, 0], G_pos_driftcorr[:, 1], G_prec, G_psf))
    column_names = ['Frame', 'pos_x/um', 'pos_y/um', 'precision/nm', 'PSF/nm']
    R_data_filtered = np.vstack((column_names, R_data_filtered))
    G_data_filtered = np.vstack((column_names, G_data_filtered))
    np.savetxt(path + f'R_data_driftcorr{Cell}.txt', R_data_filtered, fmt='%s', delimiter='\t')
    np.savetxt(path + f'G_data_driftcorr{Cell}.txt', G_data_filtered, fmt='%s', delimiter='\t')

########################################################################################################
#### Function to load the drift files 

def load_files_drift(Cell, path):
    ### Load the files from preprocessing Filtering PFO 
    R_data_driftcorr = np.loadtxt(path +f'R_data_driftcorr{Cell}.txt', skiprows=1)
    G_data_driftcorr = np.loadtxt(path +f'G_data_driftcorr{Cell}.txt', skiprows=1)
    # Extract the columns from the data array
    R_frame_driftcorr = R_data_driftcorr[:, 0]
    R_pos_driftcorr = R_data_driftcorr[:, 1:3]
    R_prec_driftcorr = R_data_driftcorr[:, 3]
    R_psf_driftcorr = R_data_driftcorr[:, 4]
    G_frame_driftcorr = G_data_driftcorr[:, 0]
    G_pos_driftcorr = G_data_driftcorr[:, 1:3]
    G_prec_driftcorr = G_data_driftcorr[:, 3]
    G_psf_driftcorr = G_data_driftcorr[:, 4]
    return R_pos_driftcorr, G_pos_driftcorr, R_prec_driftcorr, G_prec_driftcorr, R_psf_driftcorr, G_psf_driftcorr, R_frame_driftcorr, G_frame_driftcorr


def fake_drift(total_frame, R_pos, G_pos, R_frame, G_frame, drift_per_frame):
    ## USE drift corrected data 
    x_drifted = np.cumsum(np.ones((total_frame, 1)) * drift_per_frame)
    y_drifted = np.cumsum(np.ones((total_frame, 1)) * drift_per_frame)
    fake_frame = np.cumsum(np.ones((total_frame, 1)))
    emitter1 = np.vstack([fake_frame, x_drifted,y_drifted])
    emitter1 = np.transpose(emitter1)
    R_pos1, G_pos1 = drift_correction_new2(emitter1, R_pos, G_pos, R_frame, G_frame, plotting = True)
    return R_pos1, G_pos1


