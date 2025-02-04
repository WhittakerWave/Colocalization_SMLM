
#####################################################################################################
###### Main Function for colocalization 
from package_func import *
from support_func import *

#####################################################################################################
### Method I: Using Pairwise distance based on distance 
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def closest_pairs_index(R_filtered, G_filtered, R_frame, G_frame, threshold):
    ## Hard thresholding
    ## Or use cKDTree and query_ball_point
    R_tree = KDTree(R_filtered)
    G_tree = KDTree(G_filtered)
    pairs_R = G_tree.query_radius(R_filtered, threshold)
    pairs_G = R_tree.query_radius(G_filtered, threshold)
    pairs_distance = []
    for i, indices in enumerate(pairs_R):
        for j in indices:
            dist = np.linalg.norm(R_filtered[i] - G_filtered[j])
            pairs_distance.append((dist, [i, j]))
    for i, indices in enumerate(pairs_G):
        for j in indices:
            dist = np.linalg.norm(G_filtered[i] - R_filtered[j])
            pairs_distance.append((dist, [j, i]))
    pairs_distance.sort(key=lambda x: x[0])
    final_pairs = []
    pairs_index = []
    used_points_R = set()
    used_points_G = set()
    for dist, pair in pairs_distance:
         if pair[0] not in used_points_R and pair[1] not in used_points_G:
            used_points_R.add(pair[0])
            used_points_G.add(pair[1])
            final_pairs.append((dist, pair))
            pairs_index.append(pair)
    frame_pairs = []
    for pair in pairs_index:
        if R_frame is not None and G_frame is not None:
            frame_pairs.append([R_frame[pair[0]], G_frame[pair[1]]])
    return final_pairs, pairs_index, frame_pairs

def closest_pairs_index_precision(R_pos, G_pos, R_prec, G_prec, factor, threshold, R_frame, G_frame):
    ## Combine the precision
    ## Or use cKDTree and query_ball_point
    R_tree = KDTree(R_pos)
    G_tree = KDTree(G_pos)
    pairs_R = G_tree.query_radius(R_pos, threshold)
    pairs_G = R_tree.query_radius(G_pos, threshold)
    pairs_distance = []
    for i, indices in enumerate(pairs_R):
        for j in indices:
            dist = np.linalg.norm(R_pos[i] - G_pos[j])
            pairs_distance.append((dist, [i, j]))
    for i, indices in enumerate(pairs_G):
        for j in indices:
            dist = np.linalg.norm(G_pos[i] - R_pos[j])
            pairs_distance.append((dist, [j, i]))
    pairs_distance.sort(key=lambda x: x[0])
    final_pairs = []
    used_points_R = set()
    used_points_G = set()
    frame_pairs = []
    for dist, pair in pairs_distance:
        vary_thre = factor * np.sqrt(R_prec[pair[0]]**2 + G_prec[pair[1]]**2)
        if dist <= vary_thre:
            if pair[0] not in used_points_R and pair[1] not in used_points_G:
                used_points_R.add(pair[0])
                used_points_G.add(pair[1])
                if R_frame is not None and G_frame is not None:
                    frame_pairs.append([R_frame[pair[0]], G_frame[pair[1]]])
                final_pairs.append((dist, pair))
    return final_pairs, frame_pairs


import networkx as nx
print(nx.__version__)
def prob_pair(point1, point2, sigma1, sigma2, num_points, threshold):
    point1_prec = np.random.normal(loc = point1, scale = sigma1, size=(num_points, 2))
    point2_prec = np.random.normal(loc = point2, scale = sigma2, size=(num_points, 2))
    distances = np.sqrt(np.sum((point2_prec - point1_prec)**2, axis=1))
    # Count the number of points that fall within the circle of radius r
    num_points_within_circle = np.count_nonzero(distances <= threshold)
    probability = num_points_within_circle / num_points
    return probability

#####################################################################################################
##### Method II: Update new way to do colocalization
def pair_matching_max_weight_nx(R_pos, G_pos, R_prec, G_prec, d_true_thre, dis_tree_thre_factor, num_MC_points, R_frame, G_frame):
    R_tree = KDTree(R_pos)
    G_tree = KDTree(G_pos)
    ### new maxifum dis_tree_thre 
    dis_tree_thre = np.max(np.concatenate([R_prec, G_prec]))*np.sqrt(2)*dis_tree_thre_factor
    pairs_R = G_tree.query_radius(R_pos, dis_tree_thre)
    pairs_G = R_tree.query_radius(G_pos, dis_tree_thre)
    pairs_distance = []
    for i, indices in enumerate(pairs_R):
        for j in indices:
            dist = np.linalg.norm(R_pos[i] - G_pos[j])
            if dist/np.sqrt(R_prec[i]**2 + G_prec[j]**2) <= dis_tree_thre_factor:
                pairs_distance.append((dist, [i, j]))
    for i, indices in enumerate(pairs_G):
        for j in indices:
            dist = np.linalg.norm(G_pos[i] - R_pos[j])
            if dist/np.sqrt(G_prec[i]**2 + R_prec[j]**2) <= dis_tree_thre_factor:
                pairs_distance.append((dist, [j, i]))
    weights_matrix = np.zeros((len(R_pos), len(G_pos)))
    for _, item in pairs_distance:
        item[0] = int(item[0])
        item[1] = int(item[1])
        prob = prob_pair(R_pos[item[0]], G_pos[item[1]], R_prec[item[0]], G_prec[item[1]], num_MC_points, d_true_thre)
        weights_matrix[item[0], item[1]] = prob
    # create an empty bipartite graph
    # weights_matrix[~mask] = 0
    weights_matrix = np.array(weights_matrix*100000).astype(int)
    G = nx.Graph()
    # add the vertices from each partition
    G.add_nodes_from(range(len(R_pos)), bipartite=0)
    G.add_nodes_from(range(len(R_pos), len(R_pos)+len(G_pos)), bipartite=1)
    # add the weighted edges between the vertices that need to be matched
    for i in range(len(R_pos)):
        for j in range(len(G_pos)):
            if weights_matrix[i,j]!= 0:
                G.add_edge(i, j+len(R_pos), weight = weights_matrix[i,j])
    # compute the maximum weight matching
    matching = nx.max_weight_matching(G, maxcardinality=False, weight='weight')
    left_nodes = set(n for n, d in G.nodes(data=True) if d['bipartite']==0)
    right_nodes = set(G) - left_nodes
    matched_edges = [(i, j) if i in left_nodes else (j, i) for i, j in matching]
    ## renormalized the second coordinate to j-len(R_pos)
    matched_edges1 = [(i, j-len(R_pos)) for i, j in matched_edges]
    ### Added the frame pairs
    frame_pairs = []
    for pair in matched_edges1:
        if R_frame is not None and G_frame is not None:
            frame_pairs.append([R_frame[pair[0]], G_frame[pair[1]]])
    return  matched_edges1, frame_pairs

#####################################################################################################
######## Main function, iterative Monte Carlo to estimate the number of pairs
def run_exp_iterative_MC(R_pos, G_pos, R_Prec, G_Prec, R_frame, G_frame, area, number_out_iter, number_in_iter, frame_len, path, cell, filedate, d_true_thre, dis_tree_thre_factor, num_MC_points):
    ### Calculate the first pairing result
    pair_exp, frame_pairs = pair_matching_max_weight_nx(R_pos, G_pos, R_Prec, G_Prec, \
            d_true_thre = d_true_thre, dis_tree_thre_factor = dis_tree_thre_factor, num_MC_points = num_MC_points, \
            R_frame = R_frame, G_frame = G_frame)
    # plot_coloca(pair_exp, R_pos, G_pos, R_Prec, G_Prec, name = None)
    # Create the directories if they don't exist
    output_directory1 = os.path.join(path, f"coloca_results{filedate}/Frame_RG_Initial_Pairs_Region")
    output_directory2 = os.path.join(path, f"coloca_results{filedate}/Initial_RG_pair_index_Region")
    ## Make directory 
    os.makedirs(output_directory1, exist_ok=True)
    os.makedirs(output_directory2, exist_ok=True)
    # Save the files
    file_path1 = os.path.join(output_directory1, f"Frame_RG_Intial_Pairs_for_{cell}.txt")
    file_path2 = os.path.join(output_directory2, f"Initial_RG_pair_index_for_{cell}.txt")
    np.savetxt(file_path1, frame_pairs, delimiter=",", fmt='%s', header = "R data, G data")
    np.savetxt(file_path2, pair_exp, delimiter=",", header = 'R index, G index')
    ## Plot the frame difference
    frame_diff(frame_pairs, True, cell = cell, path_save = os.path.join(path, f'coloca_results{filedate}/Frame_Diff_Region/')) 
    num_R_initial = len(R_pos)
    num_G_initial = len(G_pos)
    density_R_initial = num_R_initial/area
    density_G_initial = num_G_initial/area
    num_R = len(R_pos)
    num_G = len(G_pos)
    MC_hist = []
    pair_est = []
    j = 0
    while j < number_out_iter:
        print(f'Outer Iteration: {j}')
        j = j + 1
        RG_overlap_pair_simu_list = []
        for _ in range(number_in_iter):
            R_pos_simu, G_pos_simu, R_prec_list, G_prec_list, R_frame_simu, G_frame_simu = \
                homo_Possion_Process1(num_R, num_G, R_Prec, G_Prec, area, frame_len)
            # R_prec_list = np.array([num / 1000 for num in R_prec_list])
            # G_prec_list = np.array([num / 1000 for num in G_prec_list])
            if len(R_pos_simu)==0 or len(G_pos_simu)==0:
                RG_overlap_pair_simu_list.append(0)
            else:
                RG_overlap_pair_simu, _ = pair_matching_max_weight_nx(R_pos_simu, G_pos_simu, R_prec_list, G_prec_list, \
                    d_true_thre = d_true_thre, dis_tree_thre_factor = dis_tree_thre_factor,  num_MC_points = num_MC_points, \
                    R_frame = R_frame_simu, G_frame = G_frame_simu)
                RG_overlap_pair_simu_list.append(len(RG_overlap_pair_simu))
        ## Calculate the average of the overlap list from MC simulation given the num_R and num_G
        RG_overlap_pair_simu_ave = int(np.mean(RG_overlap_pair_simu_list))
        ## Append the MC for RG overlap pair 
        MC_hist.append(RG_overlap_pair_simu_ave)
        pair_est.append(len(pair_exp) - RG_overlap_pair_simu_ave)
        num_R = num_R_initial - (len(pair_exp) - RG_overlap_pair_simu_ave)
        num_G = num_G_initial - (len(pair_exp) - RG_overlap_pair_simu_ave)
    # Names of the variables
    data_names = ['num_R in the subregion', 'num_G in the subregion', 'Effective Area', 'Density of R', 'Density of G', 'Pair Exp Initial Found', \
                'Number of MC Steps', 'Number of MC in each step', 'Monte Carlo History', \
                'True Pair Estimate', 'Number of MC Points in est prob','True Distance Threshold/nm', 'Distance Tree Threshold Factor']  
    ### Values of the variables to save
    data_values = [int(num_R_initial), int(num_G_initial), float(area), float(density_R_initial), float(density_G_initial), \
                  float(len(pair_exp)), int(number_out_iter), int(number_in_iter), \
                   np.asarray(MC_hist, dtype=float), np.asarray(pair_est, dtype=float), int(num_MC_points), \
                   d_true_thre*1000, dis_tree_thre_factor]  
    data = list(zip(data_names, data_values))  # Combine names and values as pairs
    # Save the data to the text file

    output_directory3 = os.path.join(path, f"coloca_results{filedate}/Num_RG_Overlap_Region")
    os.makedirs(output_directory3, exist_ok=True)
    file_path3 = os.path.join(output_directory3, f"Num_RG_overlap_for_{cell}.txt")
    with open(file_path3, "w") as file:
        for name, value in data:
            file.write(f"{name}: {value}\n")
    ### Return the length of pairs, MC history and pair estimation history
    return len(pair_exp), MC_hist, pair_est


##### Baseline I: Fixed some Threshold (e.g., 50 nm), find how many pairs lie within this distance
######## Main function, iterative Monte Carlo to estimate the number of pairs
def run_exp_iterative_MC_fixed_thre(R_pos, G_pos, R_Prec, G_Prec, R_frame, G_frame, threshold, area, number_out_iter, number_in_iter, frame_len, path, cell, filedate, num_MC_points):
    ### Calculate the first pairing result
    _, pair_exp, frame_pairs = closest_pairs_index(R_pos, G_pos, R_frame, G_frame, threshold)
    # plot_coloca(pair_exp, R_pos, G_pos, R_Prec, G_Prec, name = None)
    # Create the directories if they don't exist
    output_directory1 = os.path.join(path, f"coloca_results{filedate}/Frame_RG_Initial_Pairs_Region")
    output_directory2 = os.path.join(path, f"coloca_results{filedate}/Initial_RG_pair_index_Region")
    os.makedirs(output_directory1, exist_ok=True)
    os.makedirs(output_directory2, exist_ok=True)
    # Save the files
    file_path1 = os.path.join(output_directory1, f"Frame_RG_Intial_Pairs_for_{cell}.txt")
    file_path2 = os.path.join(output_directory2, f"Initial_RG_pair_index_for_{cell}.txt")
    np.savetxt(file_path1, frame_pairs, delimiter=",", fmt='%s', header = "R data, G data")
    np.savetxt(file_path2, pair_exp, delimiter=",", header = 'R index, G index')
    ## Plot the frame difference
    frame_diff(frame_pairs, True, cell = cell, path_save = os.path.join(path, f'coloca_results{filedate}/Frame_Diff_Region/')) 
    num_R_initial = len(R_pos)
    num_G_initial = len(G_pos)
    density_R_initial = num_R_initial/area
    density_G_initial = num_G_initial/area
    num_R = len(R_pos)
    num_G = len(G_pos)
    MC_hist = []
    pair_est = []
    j = 0
    while j < number_out_iter:
        print(f'Iteration: {j}')
        j = j + 1
        RG_overlap_pair_simu_list = []
        for _ in range(number_in_iter):
            R_pos_simu, G_pos_simu, R_prec_list, G_prec_list, R_frame_simu, G_frame_simu = \
                homo_Possion_Process1(num_R, num_G, R_Prec, G_Prec, area, frame_len)
            # R_prec_list = np.array([num / 1000 for num in R_prec_list])
            # G_prec_list = np.array([num / 1000 for num in G_prec_list])
            if len(R_pos_simu)==0 or len(G_pos_simu)==0:
                RG_overlap_pair_simu_list.append(0)
            else:
                _, RG_overlap_pair_simu, frame_pairs = closest_pairs_index(R_pos_simu, G_pos_simu, R_frame = R_frame_simu,\
                                G_frame = G_frame_simu, threshold = threshold)
                RG_overlap_pair_simu_list.append(len(RG_overlap_pair_simu))
        ## Calculate the average of the overlap list from MC simulation given the num_R and num_G
        RG_overlap_pair_simu_ave = int(np.mean(RG_overlap_pair_simu_list))
        ## Append the MC for RG overlap pair 
        MC_hist.append(RG_overlap_pair_simu_ave)
        pair_est.append(len(pair_exp) - RG_overlap_pair_simu_ave)
        num_R = num_R_initial - (len(pair_exp) - RG_overlap_pair_simu_ave)
        num_G = num_G_initial - (len(pair_exp) - RG_overlap_pair_simu_ave)
    # Names of the variables
    data_names = ['num_R in the subregion', 'num_G in the subregion', 'Effective Area', 'Density of R', 'Density of G', 'Pair Exp Initial Found', \
                'Number of MC Steps', 'Number of MC in each step', 'Monte Carlo History', \
                'True Pair Estimate', 'Number of MC Points in est prob','True Distance Threshold/nm', 'Distance Tree Threshold Factor']  
    ### Values of the variables to save
    data_values = [int(num_R_initial), int(num_G_initial), float(area), float(density_R_initial), float(density_G_initial), \
                  float(len(pair_exp)), int(number_out_iter), int(number_in_iter), \
                   np.asarray(MC_hist, dtype=float), np.asarray(pair_est, dtype=float), int(num_MC_points)]  
    data = list(zip(data_names, data_values))  # Combine names and values as pairs
    # Save the data to the text file
    output_directory3 = os.path.join(path, f"coloca_results{filedate}/Num_RG_Overlap_Region")
    os.makedirs(output_directory3, exist_ok=True)
    file_path3 = os.path.join(output_directory3, f"Num_RG_overlap_for_{cell}.txt")
    with open(file_path3, "w") as file:
        for name, value in data:
            file.write(f"{name}: {value}\n")
    #np.savetxt(path + f"/coloca_results{filedate}/Num_RG_overlap_Region/Num_RG_overlap_for_{cell}.txt", data, delimiter=",", fmt='%s')
    return len(pair_exp), MC_hist, pair_est