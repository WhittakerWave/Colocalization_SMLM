
from package_func import *
from filter_pre import *
####################################################################
##### function to study the regional statistics 

### Function to study the colocalization by regions
def region_study(rows, cols, R_points_intersect, G_points_intersect, R_prec_intersect, G_prec_intersect, factor, item):
    RG_region_overlap = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if R_points_intersect[i,j] is None or G_points_intersect[i,j] is None: 
                RG_region_overlap[i, j] = 0
                continue
            else:
                # pair = closest_pairs_index(R_points_intersect[i,j], G_points_intersect[i,j], item)
                # pair, _ = closest_pairs_index_precision(R_points_intersect[i,j],  G_points_intersect[i,j], \
                #               R_prec_intersect[i,j]/1000, G_prec_intersect[i,j]/1000, \
                #               factor, threshold = item, R_frame = None, G_frame = None)
                pair = pair_matching_max_weight(R_points_intersect[i,j], G_points_intersect[i,j], \
                        R_prec_intersect[i,j]/1000, G_prec_intersect[i,j]/1000, \
                        thre_tag = 5/1000, thre_tree = 100/1000, num_MC_points=int(1e4))
                pair_index = [row[1] for row in pair]
                if len(pair_index) == 0:
                    continue
                else:
                    plotname = f"Subregion ({i}, {j})"
                    #plot_coloca(pair_index, R_points_intersect[i,j], G_points_intersect[i,j], name = plotname)
                    print(f"Pairs in Subregion ({i}, {j}): {len(pair_index)}")
                RG_region_overlap[i, j] = len(pair_index)
    print(RG_region_overlap)


class RegionSelector:
    def __init__(self, R_data, G_data, cellname=None, width = None, height = None, path_select=None):
        self.R_data = R_data
        self.G_data = G_data
        self.cellname = cellname
        self.fig, self.ax = plt.subplots()
        self.df_R = pd.DataFrame({'x': R_data[:,0], 'y': R_data[:,1]})
        self.df_G = pd.DataFrame({'x': G_data[:,0], 'y': G_data[:,1]})
        self.width = width 
        self.height = height
        self.path_select = path_select
        self.ax = sns.kdeplot(data=self.df_R, x='x', y='y', cmap="Greens", shade=True, bw_adjust=0.5, ax=self.ax, label='HALO')
        self.ax.set_xlabel('x/um',fontsize = 20)
        self.ax.set_ylabel('y/um',fontsize = 20)
        #self.ax.set_title(f"HALO for {cellname}", fontsize = 20)
        self.ax = sns.kdeplot(data=self.df_G, x='x', y='y', cmap="Purples", shade=True, bw_adjust=0.5, ax=self.ax, label='SNAP')
        self.ax.set_xlabel('x/um',fontsize = 20)
        self.ax.set_ylabel('y/um',fontsize = 20)
        #self.ax.set_title(f"SNAP for {cellname}", fontsize = 20)
        self.rectangles = []
        self.select_points_R = []
        self.select_points_G = []
        # New list to store the indices of selected points
        self.select_indices_R = []  
        self.select_indices_G = []
        self.center_list = []
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
    
    def on_press(self, event):
        if event.button == 1:
            if event.inaxes is not None:
                ax = event.inaxes
                x, y = ax.transData.inverted().transform([event.x, event.y])
                center = (x, y)
                self.center_list.append([center[0], center[1], self.width, self.height])
                # width = width   # adjust as needed
                # height = height  # adjust as needed
                rect = Rectangle((center[0]-self.width/2, center[1]-self.height/2), self.width, self.height, \
                                linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                self.rectangles.append(rect)
                points_in_region_R = self.R_data[(self.df_R['x'] >= center[0]-self.width/2) & (self.df_R['x'] <= center[0]+self.width/2)\
                                & (self.df_R['y'] >= center[1]-self.height/2) & (self.df_R['y'] <= center[1]+self.height/2)]
                points_in_region_G = self.G_data[(self.df_G['x'] >= center[0]-self.width/2) & (self.df_G['x'] <= center[0]+self.width/2)\
                                & (self.df_G['y'] >= center[1]-self.height/2) & (self.df_G['y'] <= center[1]+self.height/2)]
                self.select_points_R.append(points_in_region_R)
                self.select_points_G.append(points_in_region_G)
                # Get the indices of selected points and store them in the selected_indices list
                indices_R = np.where((self.df_R['x'].values >= center[0] - self.width / 2) &
                                   (self.df_R['x'].values <= center[0] + self.width / 2) &
                                   (self.df_R['y'].values >= center[1] - self.height / 2) &
                                   (self.df_R['y'].values <= center[1] + self.height / 2))[0].tolist()
                indices_G = np.where((self.df_G['x'].values >= center[0] - self.width / 2) &
                                   (self.df_G['x'].values <= center[0] + self.width / 2) &
                                   (self.df_G['y'].values >= center[1] - self.height / 2) &
                                   (self.df_G['y'].values <= center[1] + self.height / 2))[0].tolist()
                self.select_indices_R.append(indices_R)
                self.select_indices_G.append(indices_G)
                # add number label to rectangle
                rect_label = len(self.rectangles) 
                ax.text(center[0], center[1], rect_label, color='r', ha='center', va='center',fontsize=20)
                self.fig.canvas.draw()
        np.savetxt(self.path_select + 'subregion_select_info.txt', np.array(self.center_list), fmt='%s', delimiter='\t')
        np.save(self.path_select +'subregion_select_indices_R.npy', self.select_indices_R)
        np.save(self.path_select +'subregion_select_indices_G.npy', self.select_indices_G)

########################################################################################################
#### function to save the subregion selection files 
def save_files_subregion(Cell, R_select_indices, G_select_indices, R_pos_driftcorr, G_pos_driftcorr, R_prec_driftcorr, G_prec_driftcorr, \
        R_psf_driftcorr, G_psf_driftcorr, R_frame_driftcorr, G_frame_driftcorr, path):
    column_names = ['Frame', 'pos_x/um', 'pos_y/um', 'precision/nm', 'PSF/nm']

    for index in range(len(R_select_indices)):
        r_index = R_select_indices[index]
        R_pos_select = R_pos_driftcorr[r_index]
        R_prec_select = R_prec_driftcorr[r_index]
        R_psf_select = R_psf_driftcorr[r_index]
        R_frame_select = R_frame_driftcorr[r_index]
   
        R_data_select = np.column_stack((R_frame_select, R_pos_select[:, 0], R_pos_select[:, 1], R_prec_select, R_psf_select))
        R_data_select = np.vstack((column_names, R_data_select))
        np.savetxt(path + f'R_data_{Cell}_subregion{index}.txt', R_data_select, fmt='%s', delimiter='\t', header='\t'.join(column_names))

    for index in range(len(G_select_indices)):
        g_index = R_select_indices[index]
        G_pos_select = G_pos_driftcorr[g_index]
        G_prec_select = G_prec_driftcorr[g_index]
        G_psf_select = G_psf_driftcorr[g_index]
        G_frame_select = G_frame_driftcorr[g_index]

        G_data_select = np.column_stack((G_frame_select, G_pos_select[:, 0], G_pos_select[:, 1], G_prec_select, G_psf_select))
        G_data_select = np.vstack((column_names, G_data_select))
        np.savetxt(path + f'G_data_{Cell}_subregion{index}.txt', G_data_select, fmt='%s', delimiter='\t', header='\t'.join(column_names))
