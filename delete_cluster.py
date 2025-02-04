

from package_func import *
from publish_format import *
from cluster_func import *
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from sklearn.cluster import DBSCAN

##### Color for plotting 
b2AR_color = tuple(c/255 for c in [0, 176, 80])
Gs_color = tuple(c/255 for c in [255, 0, 255])
complex_color = tuple(c/255 for c in [63, 83, 135])

def cluster_plot(labels, position, n_clusters_, indicator, prec):
    ## if prec.all()
    ## if prec is None, use 20 as precision for all 
    if prec is None:
        prec = 20*np.ones(len(position))
    # Plot for DBSCAN cluster
    unique_labels = set(labels)
    # colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(unique_labels))]
    # Make backgroud points small 
    fig, ax = plt.subplots()
    green_dark =  tuple(c/255 for c in [0, 82, 12])
    for k in unique_labels:
        if k == -1:
            # Green points used for noise points (backgrounds) # Whether the class member is equal to the k-th label
            class_member_mask = labels == k
            xy = position[class_member_mask]
            # 0.05 is to make background points small 
            prec_temp = prec[class_member_mask]
            if indicator == 'HALO':
                ax.scatter(xy[:, 0], xy[:, 1], facecolors='none', s = prec_temp[:],  alpha = 0.05, c = b2AR_color)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], facecolors='none', s = prec_temp[:],  alpha = 0.05, c = Gs_color)
        else: 
            if prec is None:
                prec = np.ones(len(position))
            class_member_mask = labels == k
            xy = position[class_member_mask]
            # 5 is to make the clusters big 
            prec_temp = prec[class_member_mask]
            if indicator == 'HALO':
                ax.scatter(xy[:, 0], xy[:, 1], facecolors='none', s = prec_temp[:],  alpha = 1, c = 'r')
            else:
                ax.scatter(xy[:, 0], xy[:, 1], facecolors='none', s = prec_temp[:],  alpha = 1, c = 'r')
    #### Write down how many total clusters in the title
    if indicator == 'HALO':
        ax.set_title(f"Estimated number of HALO clusters: {n_clusters_}", fontsize = 20)
    else:
        ax.set_title(f"Estimated number of SNAP clusters: {n_clusters_}", fontsize = 20)
    ax.set_xlabel(r'x/$\mu m$')
    ax.set_ylabel(r'y/$\mu m$')
    plt.show()
    '''
    useLargeSize1(plt, ax, marker_lines=None, fontsize=20, fontname=None, fontproperties=None, LW=2.3,\
                axis_fontsize=25, legend_fontsize=20)
    plt.show()
    ###########################################################################################
    # Make backgroud points precision the same as clusters
    fig, ax = plt.subplots()
    circles = []
    legend_labels = []
    for k in unique_labels:
        if k == -1:
            # Green points used for noise points (backgrounds) # Whether the class member is equal to the k-th label
            class_member_mask = labels == k
            xy = position[class_member_mask]
            prec_temp = prec[class_member_mask]
            #ax.scatter(xy[:, 0], xy[:, 1], facecolors='none', s = prec_temp[:],  alpha = 0.75, c = 'green')
            for i in range(len(xy)):
                if indicator == 'HALO':
                    circle = plt.Circle((xy[i, 0], xy[i, 1]), prec_temp[i]/1000, alpha = 0.3, color = b2AR_color)
                else:
                    circle = plt.Circle((xy[i, 0], xy[i, 1]), prec_temp[i]/1000, alpha = 0.3, color = Gs_color)
                ax.add_artist(circle)
                circles.append(circle)
                legend_labels.append('Outliers')
        else: 
            if prec is None:
                prec = np.ones(len(position))
            class_member_mask = labels == k
            xy = position[class_member_mask]
            prec_temp = prec[class_member_mask]
            #ax.scatter(xy[:, 0], xy[:, 1], facecolors='none', s = prec_temp[:],  alpha = 1, c = 'purple') 
            for i in range(len(xy)):
                if indicator == 'HALO':
                    circle = plt.Circle((xy[i, 0], xy[i, 1]), prec_temp[i]/1000, alpha = 1, color= 'r')
                else: 
                    circle = plt.Circle((xy[i, 0], xy[i, 1]), prec_temp[i]/1000, alpha = 1, color= 'r')
                ax.add_artist(circle)
                circles.append(circle)
                legend_labels.append('Cluster')
    if indicator == 'HALO':
        ax.set_title(f"Estimated number of HALO clusters: {n_clusters_}", fontsize = 22)
    else:
        ax.set_title(f"Estimated number of SNAP clusters: {n_clusters_}", fontsize = 22)
    ax.set_xlabel(r'x/$\mu m$')
    ax.set_ylabel(r'y/$\mu m$')
    ax.autoscale(True)
    ax.set_aspect('equal', 'box')
    #ax.set_xticks(range(18.8, 20.2, 0.5))
    #ax.set_yticks(range(19.5, 21, 0.5))
    ax.set_xlim([min(position[:, 0]), max(position[:, 0])])
    ax.set_ylim([min(position[:, 1]), max(position[:, 1])])
    useLargeSize1(plt, ax, marker_lines=None, fontsize=22, fontname=None, fontproperties=None, LW=2.3,\
                axis_fontsize=28, legend_fontsize=20)
    plt.show()
    '''

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

        # Initialize a list to store convex hull coordinates
        convex_hull_coordinates = []

        # Extract the coordinates of each cluster's convex hull
        for cluster_label in range(n_clusters):
            cluster_points = pos_fit[labels == cluster_label]
            # Calculate the convex hull of the cluster points
            hull = ConvexHull(cluster_points)
            # Append the coordinates of the convex hull to the list
            convex_hull_coordinates.append(cluster_points[hull.vertices])

        return labels, core_samples_mask, n_clusters, n_back, convex_hull_coordinates
    else:
        print("Not enough points for DBSCAN clustering.")
        print("DBSCAN cannot be performed due to errors.")
        return [-1]*len(pos_fit), None, 0, len(pos_fit), None  # Return None for all variables

########################################################################################
##### Delete the points in clusters as well as the corresponding points in the other side 

def delete_cluster(R_data, G_data, eps, min_samples, path):
    R_frame, R_pos, R_prec,  R_psf  = R_data[:,0], R_data[:,1:3], R_data[:,3], R_data[:,4]
    G_frame, G_pos, G_prec,  G_psf  = G_data[:,0], G_data[:,1:3], G_data[:,3], G_data[:,4]
    R_labels, _, R_n_clusters, _ , R_convex_hull_coord = dbscan_run(eps = eps, min_samples = min_samples, pos_fit = np.array(R_pos))
    G_labels, _, G_n_clusters, _ , G_convex_hull_coord = dbscan_run(eps = eps, min_samples = min_samples, pos_fit = np.array(G_pos))
    # Initialize lists to store indices
    R_indices_in_G_convex_hull = []
    G_indices_in_R_convex_hull = []
    # Create a Shapely Polygon for the convex hull of each G cluster
    G_convex_hulls = [Polygon(coords) for coords in G_convex_hull_coord]
    # Create a Shapely Polygon for the convex hull of each R cluster
    R_convex_hulls = [Polygon(coords) for coords in R_convex_hull_coord]

    # Check each R_pos in G_convex hulls 
    for G_hull in G_convex_hulls:
        for idx, point in enumerate(R_pos):
            if G_hull.contains(Point(point)):
                R_indices_in_G_convex_hull.append(idx)
    # Convert the list to a NumPy array if needed
    R_indices_in_G_convex_hull = np.array(R_indices_in_G_convex_hull)

    # Check each R_pos in G_convex hulls 
    for R_hull in R_convex_hulls:
        for idx, point in enumerate(G_pos):
            if R_hull.contains(Point(point)):
                G_indices_in_R_convex_hull.append(idx)
    # Convert the list to a NumPy array if needed
    G_indices_in_R_convex_hull = np.array(G_indices_in_R_convex_hull)

    # Get the indices of data points that belong to clusters (excluding noise points)
    R_cluster_indices = np.where(R_labels != -1)[0]
    G_cluster_indices = np.where(G_labels != -1)[0]

    R_indices_delete = np.concatenate((R_cluster_indices, R_indices_in_G_convex_hull))
    G_indices_delete = np.concatenate((G_cluster_indices, G_indices_in_R_convex_hull))
   
    R_indices_delete_unique = np.unique(R_indices_delete).astype(int)
    G_indices_delete_unique = np.unique(G_indices_delete).astype(int)
   
    np.savetxt(path + 'R_delete_indice_cluster.txt', R_indices_delete_unique, fmt='%d')
    np.savetxt(path + 'G_delete_indice_cluster.txt', G_indices_delete_unique, fmt='%d')
    cluster_plot(R_labels, R_pos, R_n_clusters, indicator='HALO', prec=R_prec)
    cluster_plot(G_labels, G_pos, G_n_clusters, indicator='SNAP', prec=G_prec)
    # Delete data points associated with clusters from the arrays
    R_pos1 = np.delete(R_pos, R_indices_delete_unique, axis=0)
    G_pos1 = np.delete(G_pos, G_indices_delete_unique, axis=0)
    R_prec1 = np.delete(R_prec, R_indices_delete_unique)
    G_prec1 = np.delete(G_prec, G_indices_delete_unique)
    R_frame1 = np.delete(R_frame, R_indices_delete_unique)
    G_frame1 = np.delete(G_frame, G_indices_delete_unique)
    R_psf1 = np.delete(R_psf, R_indices_delete_unique)
    G_psf1 = np.delete(G_psf, G_indices_delete_unique)
    print("R delete: " + str(len(R_indices_delete_unique)))
    print("G delete: " + str(len(G_indices_delete_unique)))
    print("R remain: " + str(len(R_pos1)))
    print("G remain: " + str(len(G_pos1)))
    return R_pos1, G_pos1, R_prec1, G_prec1, R_frame1, G_frame1, R_psf1, G_psf1, R_convex_hull_coord, G_convex_hull_coord



def delete_cluster_load(R_data, G_data, R_index_delete, G_index_delete):
    R_frame, R_pos, R_prec,  R_psf  = R_data[:,0], R_data[:,1:3], R_data[:,3], R_data[:,4]
    G_frame, G_pos, G_prec,  G_psf  = G_data[:,0], G_data[:,1:3], G_data[:,3], G_data[:,4]
    R_index_delete = R_index_delete.astype(int)
    G_index_delete = G_index_delete.astype(int)
    R_pos1 = np.delete(R_pos, R_index_delete, axis=0)
    G_pos1 = np.delete(G_pos, G_index_delete, axis=0)
    R_prec1 = np.delete(R_prec, R_index_delete)
    G_prec1 = np.delete(G_prec, G_index_delete)
    R_frame1 = np.delete(R_frame, R_index_delete)
    G_frame1 = np.delete(G_frame, G_index_delete)
    R_psf1 = np.delete(R_psf, R_index_delete)
    G_psf1 = np.delete(G_psf, G_index_delete)
    return R_pos1, G_pos1, R_prec1, G_prec1, R_frame1, G_frame1, R_psf1, G_psf1


    