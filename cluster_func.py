
################################################################################################### 
############ Clustering and related plotting functions 
##### For any errors or questions, contact xingchiyan22@gmail.com
###################################################################################################
from package_func import *
from support_func import *
####################################################################################################################
##### Function to calculate the statistics of the cluster see the list below 

def cluster_statistics(labels, pos, indicator):
    Result = namedtuple('Result', ['centroid_list', 'radius_list', 'number_inside_list', 'number_total_list', \
    'average_radius', 'pair_distance', 'area', 'area_region', 'density_cluster', 'total_inside', 'total_region', 'number_cluster', \
        'average_number_inside', 'average_number_region', 'percentage_in_cluster'])
    if max(labels)==-1:
        return Result([], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    centroid_list = []
    ## radius of each cluster
    radius_list = []
    ## Number of molecules in each cluster
    number_inside_list = []
    ## Number of molecules in certain region (2*2) outside the cluster
    number_region_list = []
    area_region = []
    for j in set(labels) - {-1}: 
        ## Find positions in each clusters
        cluster_j = pos[labels == j]
        # cluster_j[:,0:2]
        try:
            hull_j = ConvexHull(cluster_j[:,0:2])
        except scipy.spatial._qhull.QhullError:
            print("Not enough points to construct initial simplex. Skipping cluster statistics.")
            continue  #
        ### the area of the convex hull is not .area but convex hull. volume
        ##### Old way to calculate area using convex hull 
        ## area.append(hull_j.volume)
        vertices = cluster_j[hull_j.vertices]
        # cx = np.mean(vertices.T[0])
        # cy = np.mean(vertices.T[1])
        cx = np.mean(cluster_j[:, 0])
        cy = np.mean(cluster_j[:, 1])
        centroid_list.append([cx, cy])
        radius = np.max(np.linalg.norm(cluster_j[:, 0:2] - [cx, cy], axis=1))
        area_region.append(np.pi*radius**2)
        ## radius of the cluster is defined by
        ## the maximum distance of the centroid to all the points in the cluster
        # distance = np.linalg.norm([cx, cy] - cluster_j[:,0:2], axis=1)
        # radius = np.max(distance)
        # radius_list.append(radius)
        ## Use effective area to define the radius pi r*2, should use volume
        ### Old way to calculate radius
        #radius = np.sqrt(hull_j.volume/math.pi)
        radius_list.append(radius)
        number_inside_list.append(len(pos[labels == j]))
        ### The third means in a box 2*2 um^2, MAY change the number 
        count = count_points_within_box([cx, cy], pos, 2)
        number_region_list.append(count)
    area = np.mean(area_region)
    ## Calculate the average radius and pairwise distance between radius 
    total_inside = np.sum(number_inside_list)
    total_region = np.sum(number_region_list)
    average_radius = np.mean(radius_list)
    ## Calculate the pairwise distance between the centroids
    pair_distance = pdist(centroid_list, metric='euclidean')
    pair_distance = pair_distance.tolist()
    density_cluster = np.array(number_inside_list)/area_region
    density_cluster = density_cluster.tolist()

    number_cluster = len(set(labels) - {-1})
    average_number_inside = np.mean(number_inside_list)
    average_number_region = np.mean(number_region_list)
    percentage_in_cluster = np.mean(total_inside/total_region)

    # cluster_statistics_plot(radius_list, indicator, pair_distance, number_inside_list, number_total_list, area)
    return Result(centroid_list, radius_list, number_inside_list, number_region_list, average_radius,\
        pair_distance, area, area_region, density_cluster, total_inside, total_region, number_cluster, 
        average_number_inside, average_number_region, percentage_in_cluster)


####################################################################################################################
#### Old way to do statistical testing on the significance of a cluster: rank tests
def sig_test1(exp_n_clusters, density_MC, exp_density, exp_labels, T, alpha):
    ### Adjusted experimental number of cluster
    exp_n_clusters_adjust = exp_n_clusters
    for i in range(exp_n_clusters):
        ## Note that we append the density_exp
        density_MC.append(exp_density[i])
        sorted_array = sorted(density_MC, reverse=True)
        index = sorted_array.index(exp_density[i])
        ### ranked test to determine whether significant or not
        ### /exp_n_clusters
        if index <= alpha*(T+1) + 1:
            exp_n_clusters_adjust = exp_n_clusters_adjust
        else:
            ### False cluster, make the corresponding labels to be -1
            exp_labels[exp_labels == i] = -1
            exp_n_clusters_adjust = exp_n_clusters_adjust -1
    return exp_labels, exp_n_clusters_adjust

####################################################################################################################
#### New way to do statistical testing on the significance of a cluster
#### Benjamini–Hochberg procedure controls False discovery rate (more clusters will remain compared with rank test)
def sig_test2(exp_n_clusters, test_statistics_MC, exp_statistics, exp_labels, T, alpha):
    ### Benjamini–Hochberg procedure controls False discovery rate
    ### Adjusted experimental number of cluster
    ### https://en.wikipedia.org/wiki/False_discovery_rate
    if exp_n_clusters == 0:
        return exp_labels, exp_n_clusters, exp_statistics
    exp_n_clusters_adjust = exp_n_clusters
    p_list = []
    exp_density_ajust = []
    for i in range(exp_n_clusters):
        count = sum(1 for x in test_statistics_MC if x > exp_statistics[i])
        p_list.append(count/(T+1))
    p_values_with_noise = [p + random.uniform(0, 1e-11) for p in p_list]
    ##  create a dictionary to store the original order of the items
    indexed_p_values = [(p_values_with_noise[i], i) for i in range(len(p_values_with_noise))]
    sorted_p_values = sorted(indexed_p_values)
    # Calculate the line
    fig, ax = plt.subplots()
    line_x = np.arange(exp_n_clusters)
    line_y = alpha * (1.0 + line_x) / exp_n_clusters
    # Create the scatter plot and add the line
    ax.scatter(np.arange(len(sorted_p_values)), [t[0] for t in sorted_p_values], s = 1)
    ax.plot(line_x, line_y, color='red')
    # Add labels and title
    ax.set_xlabel('Index')
    ax.set_ylabel('Sorted p-values')
    ax.set_title(f'Benjamini-Hochberg Procedure with False Discovery Control Level {alpha}', fontsize = 20)
    ax.legend()
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)
    plt.show()
    ####################################
    for i in range(exp_n_clusters):
        ## break the loop if the sorted p-value is greater than the threshold
        temp = i
        if sorted_p_values[i][0] > (i+1)/exp_n_clusters*alpha:
            exp_labels[exp_labels == sorted_p_values[i][1]] = -1
            exp_n_clusters_adjust = exp_n_clusters_adjust -1
            break
        else:
            exp_n_clusters_adjust = exp_n_clusters_adjust
            exp_density_ajust.append(exp_statistics[sorted_p_values[i][1]])
    ### start from temp+1 to the end of exp_n_clusters, all labels to -1 (unsignificant clusters)
    for j in range(temp + 1, exp_n_clusters):
        exp_labels[exp_labels == sorted_p_values[j][1]] = -1
        exp_n_clusters_adjust = exp_n_clusters_adjust -1
    return exp_labels, exp_n_clusters_adjust, exp_density_ajust

####################################################################################################################
#### New way to do statistical testing on the significance of a cluster based on the statistics derived from a region
def sig_test_region(exp_n_clusters, test_statistics_MC, exp_statistics, exp_labels, T, alpha):
    ### Benjamini–Hochberg procedure controls False discovery rate
    ### Adjusted experimental number of cluster
    ### https://en.wikipedia.org/wiki/False_discovery_rate
    if exp_n_clusters == 0:
        return exp_labels, exp_n_clusters, exp_statistics
    exp_n_clusters_adjust = exp_n_clusters
    p_list = []
    exp_density_ajust = []
    for i in range(exp_n_clusters):
        count = sum(1 for x in test_statistics_MC[i] if x > exp_statistics[i])
        p_list.append(count/(T+1))
    p_values_with_noise = [p + random.uniform(0, 1e-11) for p in p_list]
    ##  create a dictionary to store the original order of the items
    indexed_p_values = [(p_values_with_noise[i], i) for i in range(len(p_values_with_noise))]
    sorted_p_values = sorted(indexed_p_values)
    # Calculate the line
    fig, ax = plt.subplots()
    line_x = np.arange(exp_n_clusters)
    line_y = alpha * (1.0 + line_x) / exp_n_clusters
    # Create the scatter plot and add the line
    ax.scatter(np.arange(len(sorted_p_values)), [t[0] for t in sorted_p_values], s = 1)
    ax.plot(line_x, line_y, color='red')
    # Add labels and title
    ax.set_xlabel('Index')
    ax.set_ylabel('Sorted p-values')
    ax.set_title(f'Benjamini-Hochberg Procedure with False Discovery Control Level {alpha}', fontsize = 20)
    ax.legend()
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)
    plt.show()
    ####################################
    for i in range(exp_n_clusters):
        ## break the loop if the sorted p-value is greater than the threshold
        temp = i
        if sorted_p_values[i][0] > (i+1)/exp_n_clusters*alpha:
            exp_labels[exp_labels == sorted_p_values[i][1]] = -1
            exp_n_clusters_adjust = exp_n_clusters_adjust -1
            break
        else:
            exp_n_clusters_adjust = exp_n_clusters_adjust
            exp_density_ajust.append(exp_statistics[sorted_p_values[i][1]])
    ### start from temp+1 to the end of exp_n_clusters, all labels to -1 (unsignificant clusters)
    for j in range(temp + 1, exp_n_clusters):
        exp_labels[exp_labels == sorted_p_values[j][1]] = -1
        exp_n_clusters_adjust = exp_n_clusters_adjust -1
    return exp_labels, exp_n_clusters_adjust, exp_density_ajust

####################################################################################################################
##### Main function to run DBSCAN, use python package 
def pre_run_dbscan(eps, min_samples, pos_fit):
    try:
        DBSCAN(eps=eps, min_samples=min_samples).fit(pos_fit)
        return True  # DBSCAN can be performed without errors
    except:
        return False  # DBSCAN would result in an error

def dbscan_run(eps, min_samples, pos_fit):
    ## DBSCAN
    if pre_run_dbscan(eps, min_samples, pos_fit):
        cluster = DBSCAN(eps = eps, min_samples = min_samples).fit(pos_fit)
        core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
        core_samples_mask[cluster.core_sample_indices_] = True
        labels = cluster.labels_
        # Number of clusters in labels, ignoring background noise if present
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Number of backgrount points
        n_back = list(labels).count(-1) 
        return labels, core_samples_mask, n_clusters, n_back
    else:
        print("Not enough points for DBSCAN clustering.")
        print("DBSCAN cannot be performed due to errors.")
        return [-1]*len(pos_fit), None, 0, len(pos_fit)  # Return None for all variables


####################################################################################################################
##### RUN DBSCAN with statistical testing on the clusters, run T monte carlo times so call T_DBSCAN 
def T_DBSCAN(eps, min_samples, T, alpha, pos, S_area, indicator, cellname, filedate, path, selection_number):
    #### Running on the true sample for T Monte Carlo trivals
    exp_labels, _, exp_n_clusters, _ = dbscan_run(eps, min_samples, pos)
    exp_cluster_stat = cluster_statistics(exp_labels, pos, indicator = indicator) 
    exp_density = exp_cluster_stat.density_cluster
    exp_number_in_cluster = exp_cluster_stat.number_inside_list
    print(f'Experimental Density: {exp_density}')
    print(f'Experimental Number in Cluster: {exp_number_in_cluster}')
    ### maximum density of clusters for Monte Carlo trials
    density_MC = []
    number_in_cluster_MC = []
    for i in range(T):
        hom_pos = np.array(homo_Possion_Process1(S_area, len(pos)))
        hom_labels, _, hom_n_clusters, _ = dbscan_run(eps = eps, min_samples = min_samples, pos_fit = hom_pos)
      
        hom_cluster_stat = cluster_statistics(hom_labels, hom_pos, indicator = indicator) 
        if hom_cluster_stat.density_cluster == []:
            hom_max_density_cluster = 0
            hom_max_number_in_cluster = 0
        else:
            ## Append the maximum density
            hom_max_density_cluster = max(hom_cluster_stat.density_cluster)
        ## Append the number in cluster
            hom_max_number_in_cluster = max(hom_cluster_stat.number_inside_list)
        density_MC.append(hom_max_density_cluster)
        number_in_cluster_MC.append(hom_max_number_in_cluster)
        # print(f'Monte Carlo Simulated Density: {density_MC}')
    ## Difference statistical testing
    exp_labels_adjust, exp_n_clusters_adjust, exp_density_ajust = sig_test2(exp_n_clusters, number_in_cluster_MC, exp_number_in_cluster, exp_labels, T, alpha)
    ## exp_labels_adjust, exp_n_clusters_adjust, exp_density_ajust = sig_test2(exp_n_clusters, density_MC, exp_density, exp_labels, T, alpha)
    ### rerange the labels to [-1, 0, 1, 2, 3 ... ]
    unique_values, indices = np.unique(exp_labels_adjust, return_inverse = True)
    exp_labels_adjust = np.searchsorted(unique_values, exp_labels_adjust) - 1
    ### 
    cluster_stat_adjust = cluster_statistics(exp_labels_adjust, pos, indicator = indicator) 
    ##### Make a folder that is dependence on the selection number
    folder_name = f'Area Selection Number{selection_number}'
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # cluster_statistics_plot(radius_list, indicator, pair_distance, number_inside_list, number_total_list, area)
    with open(folder_path + f'/statistics_{indicator}_for_{cellname}_for_Selection{selection_number}.txt', 'w') as f:
        var_names = ['centroid_list', 'radius_list', 'number_inside_list', 'number_region_list', 'average_radius', 
             'pair_distance', 'area', 'density_cluster', 'total_inside', 'total_region', 'number_clusters', 
             'average_number_inside', 'average_number_region', 'percentage_in_cluster']
        for name, item in zip(var_names, cluster_stat_adjust):
            f.write(f'{name}: {item}\n')
    np.savetxt(folder_path + f'/labels_for_{indicator}_{cellname}_for_Selection{selection_number}.txt', exp_labels_adjust)
    return exp_labels_adjust, exp_n_clusters_adjust,  cluster_stat_adjust


####################################################################################################################
##### RUN sig-DBSCAN region by region 
def T_DBSCAN_region(eps, min_samples, T, alpha, pos, prec, S_area, indicator, cellname, \
            filedate, path, selection_number, box_length):
    #### Running on the true sample for T Monte Carlo trivals
    exp_labels, _, exp_n_clusters, _ = dbscan_run(eps, min_samples, pos)
    exp_cluster_stat = cluster_statistics(exp_labels, pos, indicator = indicator) 
    exp_density = exp_cluster_stat.density_cluster
    exp_number_in_cluster = exp_cluster_stat.number_inside_list
    print(f'Experimental Density: {exp_density}')
    print(f'Experimental Number in Cluster: {exp_number_in_cluster}')
    ### maximum density of clusters for Monte Carlo trials
    density_MC = np.zeros((exp_n_clusters, T))
    number_in_cluster_MC = np.zeros((exp_n_clusters, T))
    ###### For each cluster identified
    for index, center in enumerate(exp_cluster_stat.centroid_list):
        ### take a box around the center
        points_box, _ = points_within_box(center, pos, precision = prec, box_size = box_length)
        density_MC_temp = []
        number_in_cluster_MC_temp = []
        for i in range(T):
            S_area = box_length**2
            hom_pos = homo_Possion_Process1(S_area, len(points_box))
            hom_labels, _, hom_n_clusters, _ = dbscan_run(eps = eps, min_samples = min_samples, pos_fit = hom_pos)
            hom_cluster_stat = cluster_statistics(hom_labels, hom_pos, indicator = indicator) 
            if hom_cluster_stat.density_cluster == []:
                hom_max_density_cluster = 0
                hom_max_number_in_cluster = 0
            else:
                ## Append the maximum density
                hom_max_density_cluster = max(hom_cluster_stat.density_cluster)
                ## Append the number in cluster
                hom_max_number_in_cluster = max(hom_cluster_stat.number_inside_list)
            density_MC_temp.append(hom_max_density_cluster)
            number_in_cluster_MC_temp.append(hom_max_number_in_cluster)
        density_MC[index] = density_MC_temp
        number_in_cluster_MC[index] = number_in_cluster_MC_temp
        # print(f'Monte Carlo Simulated Density: {density_MC}')
    ## Difference statistical testing
    exp_labels_adjust, exp_n_clusters_adjust, exp_density_ajust = \
        sig_test_region(exp_n_clusters, number_in_cluster_MC, exp_number_in_cluster, exp_labels, T, alpha)
    #### exp_labels_adjust, exp_n_clusters_adjust, exp_density_ajust = sig_test2(exp_n_clusters, density_MC, exp_density, exp_labels, T, alpha)
    ### rerange the labels to [-1, 0, 1, 2, 3 ... ]
    unique_values, indices = np.unique(exp_labels_adjust, return_inverse = True)
    exp_labels_adjust = np.searchsorted(unique_values, exp_labels_adjust) - 1
    ### 
    cluster_stat_adjust = cluster_statistics(exp_labels_adjust, pos, indicator = indicator) 
    ##### Make a folder that is dependence on the selection number
    folder_name = f'Area Selection Number{selection_number}'
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    cluster_statistics_plot(cluster_stat_adjust.radius_list, indicator, cluster_stat_adjust.pair_distance, \
            cluster_stat_adjust.number_inside_list, cluster_stat_adjust.number_total_list, cluster_stat_adjust.area_region)

    with open(folder_path + f'/statistics_{indicator}_for_{cellname}_for_Selection{selection_number}.txt', 'w') as f:
        var_names = ['centroid_list', 'radius_list', 'number_inside_list', 'number_region_list', 'average_radius', 
             'pair_distance', 'area', 'density_cluster', 'total_inside', 'total_region', 'number_clusters', 
             'average_number_inside', 'average_number_region', 'percentage_in_cluster']
        for name, item in zip(var_names, cluster_stat_adjust):
            f.write(f'{name}: {item}\n')
    np.savetxt(folder_path + f'/labels_for_{indicator}_{cellname}_for_Selection{selection_number}.txt', exp_labels_adjust)
    return exp_labels_adjust, exp_n_clusters_adjust, cluster_stat_adjust


#############################################################################################################
#### Input: labels for R and G as well as their position
#### Find the hotspots based on the convex hull of the cluster, if the area overlap more than threshold, return as a hotspot 
def RG_hotspot_prop(R_labels, G_labels, R_positions, G_positions, threshold, path, selection_number, cellname):
    ## Find Hotspot using area proportion based method
    RG_hotspot_pair = []
    for i in set(R_labels) - {-1}:
        ## For R clusters
        R_cluster_i = R_positions[R_labels == i]
        # ConvexHull formed by R_cluster_i
        hull_R_i = ConvexHull(R_cluster_i[:,0:2])
        # Find the positions of the vertices of the Convexhull
        R_vertices = R_cluster_i[hull_R_i.vertices]
        ## Using Convexhull.volume is the same as Polygon.area 
        # V_R = hull_R_i.volume 
        # tuple of the coordinates (x, y) for the vertices of the Convexhull used to find Polygon
        test_R = [tuple(x) for x in R_vertices[:,0:2].tolist()]
        V_R_polygon = Polygon(test_R).buffer(0)
        for j in set(G_labels) - {-1}: 
            ## For G clusters
            G_cluster_j = G_positions[G_labels == j]
            hull_G_j = ConvexHull(G_cluster_j[:,0:2])
            G_vertices =  G_cluster_j[hull_G_j.vertices]
            test_G = [tuple(x) for x in G_vertices[:,0:2].tolist()]
            V_G_polygon = Polygon(test_G).buffer(0)
            intersect = V_R_polygon.intersection(V_G_polygon).area
            intersect_ratio = intersect/min(V_R_polygon.area, V_G_polygon.area)
            ## 1) threshold conditions and 2) Have to ensure that the second coordinate index also do not repeat
            if intersect_ratio >= threshold and j not in set(tup[1] for tup in RG_hotspot_pair):
                RG_hotspot_pair.append([i,j])
                break 
    
    ##### Make a folder that is dependence on the selection number
    folder_name = f'Area Selection Number{selection_number}'
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    np.savetxt(folder_path + f'/Hotspots_{cellname}_for_Selection{selection_number}.txt', RG_hotspot_pair)
    return RG_hotspot_pair 





