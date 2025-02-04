
#####################################################################################################
##### Small functions to support, such as random point generations, SNR 
from package_func import *
#####################################################################################################
def Random_Point(num_R, num_G, prec_R, prec_G, grid):
    ## Generate Random points in the domian 
    ## here we haven't used precision, maybe use later
    scale = grid[0][1]
    R_pos = np.random.random((num_R, 2))*scale
    G_pos = np.random.random((num_G, 2))*scale
    return R_pos, G_pos

#####################################################################################################
def homo_Possion_Process1(num_R, num_G, R_precision, G_precision, area, frame_len):
    grid = [[0,np.sqrt(area)], [0,np.sqrt(area)]]
    #Simulation window parameters
    xMin = 0
    xMax = grid[0][1]
    yMin = 0
    yMax = grid[0][1]
    xDelta = xMax - xMin
    yDelta = yMax - yMin
    areaTotal = xDelta*yDelta
    #Point process parameters
    lambda_R = num_R / areaTotal #intensity (ie mean density) of the Poisson process
    lambda_G = num_G / areaTotal
    #Simulate Poisson point process
    pointsNumber_R = scipy.stats.poisson(lambda_R*areaTotal).rvs() #Poisson number of points
    pointsNumber_G = scipy.stats.poisson(lambda_G*areaTotal).rvs() #Poisson number of points
    xx_R = xDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber_R,1))) + xMin #x coordinates of Poisson points
    yy_R = yDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber_R,1))) + yMin #y coordinates of Poisson points
    points_R = np.concatenate([xx_R, yy_R], axis=1)
    xx_G = xDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber_G,1))) + xMin #x coordinates of Poisson points
    yy_G = yDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber_G,1))) + yMin #y coordinates of Poisson points
    points_G = np.concatenate([xx_G, yy_G], axis=1)
    if len(points_R) <= len(R_precision):
        R_prec_list = random.sample(list(R_precision), len(points_R))
    else:
        R_prec_list1 = random.sample(list(R_precision), len(R_precision))
        R_prec_arr = np.array(list(R_precision))
        probs = R_prec_arr / np.sum(R_prec_arr)  # Normalize probabilities
        R_prec_list2 = np.random.choice(R_prec_arr, size=len(points_R)-len(R_precision), p=probs, replace=True).tolist()
        R_prec_list = R_prec_list1 + R_prec_list2
    if len(points_G) <= len(G_precision):
        G_prec_list = random.sample(list(G_precision), len(points_G))
    else:
        G_prec_list1 = random.sample(list(G_precision), len(G_precision))
        G_prec_arr = np.array(list(G_precision))
        probs = G_prec_arr / np.sum(G_prec_arr)  # Normalize probabilities
        G_prec_list2 = np.random.choice(G_prec_arr, size = len(points_G)-len(G_precision), p=probs, replace=True).tolist()
        G_prec_list = G_prec_list1 + G_prec_list2
    '''
    R_prec_arr = np.array(list(R_precision))
    probs = R_prec_arr / np.sum(R_prec_arr)  # Normalize probabilities
    R_prec_list = np.random.choice(R_prec_arr, size=len(points_R), p=probs, replace=True).tolist()
    G_prec_arr = np.array(list(G_precision))
    probs = G_prec_arr / np.sum(G_prec_arr)  # Normalize probabilities
    G_prec_list = np.random.choice(G_prec_arr, size=len(points_G), p=probs, replace=True).tolist()
    '''
    # Generate two uniform distributions
    uniform_dist1 = np.arange(0, frame_len +1)
    uniform_dist2 = np.arange(0, frame_len +1)
    R_frame = []
    G_frame = []
    for i in range(len(points_R)):
        R_frame.append(np.random.choice(uniform_dist1, size=1)[0].astype(float))
    for j in range(len(points_G)):
        G_frame.append(np.random.choice(uniform_dist2, size=1)[0].astype(float))
    
    ### Need to pertube the points with Precision 
    R_prec_array = np.array(R_prec_list).reshape(len(points_R), 1)/1000
    G_prec_array = np.array(G_prec_list).reshape(len(points_G), 1)/1000
    gaussian_array_R = np.random.normal(loc = np.zeros((len(points_R), 2)), \
            scale = R_prec_array, size=(len(points_R), 2))
    gaussian_array_G = np.random.normal(loc = np.zeros((len(points_G), 2)), \
            scale = G_prec_array, size=(len(points_G), 2))
    # add the Gaussian array to R_pos
    points_R = points_R + gaussian_array_R
    points_G = points_G + gaussian_array_G
    return points_R, points_G, R_prec_list, G_prec_list, R_frame, G_frame

#####################################################################################################
def frame_diff(frame_pairs, title, cell, path_save):
    frame_diff = []
    for item in frame_pairs:
        frame_diff.append(abs(item[0] - item[1]))
    fig, ax = plt.subplots()
    sns.histplot(frame_diff)
    # add title and axis labels
    if title == False:
        ax.set_title('Monte Carlo Simulation Frame Difference', fontsize = 20)
    else:
        ax.set_title(f'Frame Difference of {cell}', fontsize = 20)
    ax.set_xlabel('Difference of frames in a pair (abs)')
    ax.set_ylabel('Count')
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3) 
    plt.savefig(path_save + f"/frame_diff_{cell}.png") 
    plt.close(fig)
    #plt.show()

######################################################################################################
def homo_Possion_Process2(area, num_points):
    #Simulation window parameters
    xMin = 0
    xMax = np.sqrt(area)
    yMin = 0
    yMax = np.sqrt(area)
    xDelta = xMax - xMin
    yDelta = yMax - yMin
    areaTotal = xDelta*yDelta
    #Point process parameters
    lambda0 = num_points / areaTotal #intensity (ie mean density) of the Poisson process
    #Simulate Poisson point process
    pointsNumber = scipy.stats.poisson(lambda0*areaTotal).rvs() #Poisson number of points
    xx = xDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber,1))) + xMin #x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber,1))) + yMin #y coordinates of Poisson points
    points = np.concatenate([xx, yy], axis=1)
    #### Plot the HPPS:
    '''
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=2)
    ax.set_title("Homogeneous Poisson Point Process", fontsize = 20)
    ax.set_xlabel("x/$\mu m$")
    ax.set_ylabel("y/$\mu m$")
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)
    plt.show()
    '''
    return points

#####################################################################################################
##### Function to plot the density of the data
def density_heatmap(pos, cellname, indicator):
    fig, ax = plt.subplots()
    df = pd.DataFrame({'x': pos[:,0], 'y': pos[:,1]})
    # Create a kernel density estimate plot using seaborn
    if indicator == 'HALO':
        ## cmap="Greens", 'viridis' 'Plasma', 'Inferno', "Purples"
        ## cbar_kws={'label': 'Probability Density'}
        ax = sns.kdeplot(data=df, x='x', y='y', cmap='viridis', shade=True, bw_adjust=0.5,\
                 cbar=True)
    else:
        ax = sns.kdeplot(data = df, x='x', y='y', cmap = 'viridis', shade=True, bw_adjust=0.5,\
                 cbar=True)
    ax.set_xlabel(r"x/$\mu$m")
    ax.set_ylabel(r"y/$\mu$m")
    ax.set_title(f"Probability Density Heatmap of {indicator} for {cellname}", fontsize = 20)
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3) 
    plt.show()

###############################################################################################################
## Plot the Number of Photon with Precisions
def photon_precision_plot(photon, precision, cellname, indicator):
    fig, ax = plt.subplots()
    plt.hist2d(photon, precision ,bins=100)
    ax.set_xlabel("Number Photons")
    ax.set_ylabel("Precision/nm")
    ax.set_title(f'Precision vs Number of Photon of {indicator} for {cellname}', fontsize=19)
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)
    plt.show()

#photon_precision_plot(R_data[:,7], R_data[:,6], cellname = Cell, indicator = 'HALO')
#photon_precision_plot(G_data[:,7], G_data[:,6], cellname = Cell, indicator = 'SNAP')

#####################################################################################################
def intensity_plot(intensity, cellname, indicator):
    fig, ax = plt.subplots()
    sns.histplot(intensity ,bins=100)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Counts")
    ax.set_title(f'Number of Photon of {indicator} for {cellname}', fontsize=19)
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)
    plt.show()

#intensity_plot(R_data[:,7], cellname = Cell, indicator = 'HALO')
#intensity_plot(G_data[:,7], cellname = Cell, indicator = 'HALO')

###############################################################################################################
## Plot the precision distribution (may choose a threshold for precision):
def precision_plot(prec, indicator, cellname):
    fig, ax = plt.subplots()
    sns.histplot(prec)
    ax.set_xlabel("Precision/nm")
    ax.set_ylabel("Counts")
    ax.set_title(f'{indicator} Precision Histogram of {cellname}', fontsize=19)
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)
    plt.show()

#precision_plot(R_data[:, 6], indicator = 'HALO', cellname = Cell)
#precision_plot(G_data[:, 6], indicator = 'SNAP', cellname = Cell)

###############################################################################################################
def psf_plot(psf, indicator, cellname):
    # Plot the R/G Frame histograms
    fig, ax = plt.subplots()
    sns.histplot(psf, bins = 200)
    ax.set_title(f"{indicator} PSF half width for {cellname}", fontsize = 20)
    ax.set_xlabel(f"{indicator} PSF half width/nm")
    ax.set_ylabel("Counts")
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3) 
    plt.show() 

#psf_plot(R_PSF[R_PSF < 300], indicator = 'HALO', cellname = Cell)
#psf_plot(G_PSF[G_PSF < 300], indicator = 'SNAP', cellname = Cell)

############################################################################################################
###### Calculate the average signal to noise ratio of the data
def SNR(num_photon_list, psf_list, back_var_list, pixel_size_list):
    # Convert elements to numeric type (float)
    back_var_list = back_var_list.astype(float)
    temp = np.array(psf_list) / np.array(pixel_size_list)
    snr_list = num_photon_list / (2 * temp * np.sqrt(back_var_list))
    mean_snr = np.mean(snr_list)
    return mean_snr

############################################################################################################
###### Calculate the average signal to noise ratio of the data, only calculating the nonzero entries
def SNR_nonzero(num_photon_list, psf_list, back_var_list, pixel_size_list):
    # Convert elements to numeric type (float)
    back_var_list = back_var_list.astype(float)
    temp = np.array(psf_list) / np.array(pixel_size_list)
    # Filter out zero values in temp and back_var_list
    non_zero_indices = (temp != 0) & (back_var_list != 0) \
        & (~np.isnan(num_photon_list.astype(float))) & (~np.isnan(temp.astype(float))) & (~np.isnan(back_var_list.astype(float)))
    num_photon_list = num_photon_list[non_zero_indices]
    temp = temp[non_zero_indices]
    back_var_list = back_var_list[non_zero_indices]
    snr_list = num_photon_list / (2 * temp * np.sqrt(back_var_list))
    mean_snr = np.mean(snr_list)
    return mean_snr

#####################################################################################################
## Function to plot colocalization and non-colocalization 
# from matplotlib.legend_handler import PathCollectionHandler
def plot_coloca(pair_index, R_pos, G_pos, filtered_R_Prec, filtered_G_Prec, name):
    index_R_C = np.array(pair_index).T[0]
    index_G_C = np.array(pair_index).T[1]
    
    total_index_R = list(range(0, len(R_pos)))
    total_index_G = list(range(0, len(G_pos)))
    
    index_R_NoC = [x for x in total_index_R if x not in index_R_C]
    index_G_NoC = [x for x in total_index_G if x not in index_G_C]

    R_pos_C = R_pos[index_R_C]
    R_pos_NoC = R_pos[index_R_NoC]
    R_prec_C = filtered_R_Prec[index_R_C]/1000
    R_prec_NoC = filtered_R_Prec[index_R_NoC]/1000
    
    G_pos_C = G_pos[index_G_C]
    G_pos_NoC = G_pos[index_G_NoC]
    G_prec_C = filtered_G_Prec[index_G_C]/1000
    G_prec_NoC = filtered_G_Prec[index_G_NoC]/1000

    fig, ax = plt.subplots()
    color = ['tab:green', 'tab:purple', 'tab:blue', 'tab:blue']
    b2AR_color = tuple(c/255 for c in [0, 176, 80])
    Gs_color = tuple(c/255 for c in [255, 0, 255])
    complex_color = tuple(c/255 for c in [63, 83, 135])
    
    circles = []
    legend_labels = []
    for i in range(len(R_prec_NoC)):
        circle = plt.Circle((R_pos_NoC[i, 0], R_pos_NoC[i, 1]), R_prec_NoC[i], color=b2AR_color)
        ax.add_artist(circle)
        circles.append(circle)
        legend_labels.append('HALO No Colocalization')

    for i in range(len(G_prec_NoC)):
        circle = plt.Circle((G_pos_NoC[i, 0], G_pos_NoC[i, 1]), G_prec_NoC[i], color=Gs_color)
        ax.add_artist(circle)
        circles.append(circle)
        legend_labels.append('SNAP No Colocalization')

    for i in range(len(R_pos_C)):
        circle = plt.Circle((R_pos_C[i, 0], R_pos_C[i, 1]), R_prec_C[i], color=complex_color)
        ax.add_artist(circle)
        circles.append(circle)
        legend_labels.append('HALO-SNAP Colocalization')

    for i in range(len(G_pos_C)):
        circle = plt.Circle((G_pos_C[i, 0], G_pos_C[i, 1]), G_prec_C[i], color=complex_color)
        ax.add_artist(circle)
        circles.append(circle)
        legend_labels.append('HALO-SNAP Colocalization')

    unique_labels = {}
    for i in range(len(circles)):
        if legend_labels[i] not in unique_labels:
            unique_labels[legend_labels[i]] = circles[i]
    ax.legend(unique_labels.values(), unique_labels.keys())
    
    ## 
    ax.plot([R_pos_C[:,0],G_pos_C[:,0]], [R_pos_C[:,1],G_pos_C[:,1]], color=color[3])
    ax.set_xlabel(r"x/$\mu m$")
    ax.set_ylabel(r"y/$\mu m$")
    if name is not None:
        ax.set_title(name, fontsize = 20)
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)
    plt.show()

