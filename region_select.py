
from package_func import *
#matplotlib.use('TkAgg')

### Colors 
b2AR_color = tuple(c/255 for c in [0, 176, 80])
Gs_color = tuple(c/255 for c in [255, 0, 255])
complex_color = tuple(c/255 for c in [63, 83, 135])

class RegionSelector:
    def __init__(self, data, indicator=None, cellname=None):
        self.data = data
        self.indicator = indicator
        self.cellname = cellname
        self.fig, self.ax = plt.subplots()
        self.df = pd.DataFrame({'x': data[:,0], 'y': data[:,1]})
        if self.indicator == 'HALO':
            self.ax = sns.kdeplot(data=self.df, x='x', y='y', cmap= 'viridis', shade=True, bw_adjust=0.5, ax=self.ax)
            self.ax.set_xlabel('x/um',fontsize = 20)
            self.ax.set_ylabel('y/um',fontsize = 20)
            self.ax.set_title(f"{indicator} for {cellname}", fontsize = 20)
        else:
            self.ax = sns.kdeplot(data=self.df, x='x', y='y', cmap = 'viridis', shade=True, bw_adjust=0.5, ax=self.ax)
            self.ax.set_xlabel('x/um',fontsize = 20)
            self.ax.set_ylabel('y/um',fontsize = 20)
            self.ax.set_title(f"{indicator} for {cellname}", fontsize = 20)
        self.rectangles = []
        self.selected_points = []
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
    def on_press(self, event):
        if event.button == 1:
            if event.inaxes is not None:
                ax = event.inaxes
                x, y = ax.transData.inverted().transform([event.x, event.y])
                center = (x, y)
                width = 5  # adjust as needed
                height = 5  # adjust as needed
                rect = Rectangle((center[0]-width/2, center[1]-height/2), width, height, \
                                linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                self.rectangles.append(rect)
                points_in_region = self.data[(self.df['x'] >= center[0]-width/2) & (self.df['x'] <= center[0]+width/2)\
                                & (self.df['y'] >= center[1]-height/2) & (self.df['y'] <= center[1]+height/2)]
                self.selected_points.append(points_in_region)
                # add number label to rectangle
                rect_label = len(self.rectangles) 
                ax.text(center[0], center[1], rect_label, color='r', ha='center', va='center',fontsize=20)
                self.fig.canvas.draw()

#############################################################################################################
##### Function to select certain areas 
## Define a callback function for the PolygonSelector
def onselect(verts, R_pos, G_pos):
    path = Path(verts)
    inside1 = path.contains_points(np.column_stack((R_pos[:,0], R_pos[:,1])))
    inside2 = path.contains_points(np.column_stack((G_pos[:,0], G_pos[:,1])))
    count_R = np.sum(inside1)
    count_G = np.sum(inside2)
    #filtered_R = np.column_stack((R_pos[:,0][inside1], R_pos[:,1][inside1]))
    #filtered_G = np.column_stack((G_pos[:,0][inside2], G_pos[:,1][inside2]))
    select_R_idx = np.where(inside1)[0]
    select_G_idx = np.where(inside2)[0]
    select_R = R_pos[select_R_idx, :]
    select_G = G_pos[select_G_idx, :]
    poly = Polygon(verts)
    area = poly.area
    print("Number of R points inside:", count_R)
    print("Number of G points inside:", count_G)
    print("Area of the region:", area)
    return select_R, select_G, select_R_idx, select_G_idx, area

def area_selection(R_pos, G_pos, Cell, filedate, path):
    ## Select certain area of a cell by hand 
    ## Create a scatter plot of the selected points
    fig, ax = plt.subplots()
    ax.scatter(R_pos[:,0], R_pos[:,1], s=2, c = b2AR_color, label='HALO')
    ax.scatter(G_pos[:,0], G_pos[:,1], s=2, c = Gs_color, label='SNAP')
    ax.legend()
    ax.set_title(f"{Cell}", fontsize = 20)
    ax.set_xlabel(r"x/$\mu$m")
    ax.set_ylabel(r"y/$\mu$m")
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)  
    # Create a PolygonSelector object and connect it to the callback function
    poly = PolygonSelector(ax, onselect)
    plt.show()
    _, _, select_R_idx, select_G_idx, area = onselect(poly.verts, R_pos = R_pos, G_pos = G_pos)
    info = [len(R_pos), len(G_pos), area, len(select_R_idx), len(select_G_idx)]
    info_names = ['R original number', 'G original number', 'area', 'select_R_number', 'select_G_number']
    with open(path + f'Output_files{filedate}/first_select_info_cell{Cell}.txt', 'w') as f:
        for name, value in zip(info_names, info):
            print(f'{name}: {value}', file=f)
    # Save the selected index as NumPy binary files
    np.savetxt(path + f'Output_files{filedate}/first_select_vertex_cell{Cell}.txt', poly.verts)
    np.savetxt(path + f'Output_files{filedate}/R_select_idx_cell{Cell}.txt', select_R_idx)
    np.savetxt(path + f'Output_files{filedate}/G_select_idx_cell{Cell}.txt', select_G_idx)

#################################################################################################
##### Function to select certain areas and divide into subregions 
# Define a callback function for the PolygonSelector
def oneselect_subregions(verts, cols, rows, R_pos, G_pos):
    path = Path(verts)
    inside1 = path.contains_points(np.column_stack((R_pos[:,0], R_pos[:,1])))
    inside2 = path.contains_points(np.column_stack((G_pos[:,0], G_pos[:,1])))
    count_R = np.sum(inside1)
    count_G = np.sum(inside2)
    select_R = np.column_stack((R_pos[:,0][inside1], R_pos[:,1][inside1]))
    select_G = np.column_stack((G_pos[:,0][inside2], G_pos[:,1][inside2]))
    poly = Polygon(verts)
    area = poly.area
    print("Number of R points inside:", count_R)
    print("Number of G points inside:", count_G)
    print("Area of the region:", area)
    ## Create a 2D grid of evenly spaced points using cols and rows defined above
    x, y = np.meshgrid(np.linspace(poly.bounds[0], poly.bounds[2], cols+1), np.linspace(poly.bounds[1], poly.bounds[3], rows+1))
    ## Plot the grids boundary  
    x_grid = np.linspace(poly.bounds[0], poly.bounds[2], cols+1)
    y_grid = np.linspace(poly.bounds[1], poly.bounds[3], rows+1)
    xx, yy = np.meshgrid(x_grid, y_grid)
    for i in range(cols + 1):
        plt.plot(xx[:, i], yy[:, i], '-', color='blue')
    for i in range(rows + 1):
        plt.plot(xx[i, :], yy[i, :], '-', color='blue')
    plt.show()
    ## Initialize an array to store the area of intersection
    intersection_area = np.zeros((rows, cols))
    ## Iterate over the grid cells
    for i in range(rows):
        for j in range(cols):
            ## Find the coordinates of the grid cell
            x1, x2, y1, y2 = x[i, j], x[i, j + 1], y[i, j], y[i + 1, j]
            grid_poly = box(x1, y1, x2, y2) # Create a polygon for the grid cell
            intersection = grid_poly.intersection(poly) # Find the intersection of the grid cell with the selected area
            if intersection.is_empty: # if the intersection is empty, the grid cell is outside of the selected area
                intersection_area[i, j] = 0
            else:
                intersection_area[i, j] = intersection.area # otherwise, store the area of the intersection
    histR, _, _ = np.histogram2d(select_R[:,0], select_R[:,1], bins=(rows, cols))
    histG, _, _ = np.histogram2d(select_G[:,0], select_G[:,1], bins=(rows, cols))
    # count_R_intersection = np.where(intersection_area >0, histR, 0)
    # count_G_intersection = np.where(intersection_area >0, histG, 0)
    # Create 2D array of grid cell indices
    # Use digitize() to map red points to grid cell indices
    grid_x_bounds = np.linspace(poly.bounds[0], poly.bounds[2], cols+1)
    grid_y_bounds = np.linspace(poly.bounds[1], poly.bounds[3], rows+1)
    R_x_indices = np.digitize(select_R[:, 0], grid_x_bounds)
    R_y_indices = np.digitize(select_R[:, 1], grid_y_bounds)
    R_indices = np.column_stack((R_x_indices, R_y_indices))
    # Iterate over the red points and add their coordinates to the corresponding grid cell
    R_points_intersect = np.empty((cols, rows), dtype=object)
    R_points_intersect_index = np.empty((cols, rows), dtype=object)
    for i, index in enumerate(R_indices):
        temp = np.where((R_pos == select_R[i]).all(axis=1))[0][0]
        if R_points_intersect[index[0]-1, index[1]-1] is None:
            R_points_intersect[index[0]-1, index[1]-1] = np.array([select_R[i]])
            R_points_intersect_index[index[0]-1, index[1]-1] = np.array([temp])
        else:
            R_points_intersect[index[0]-1, index[1]-1] = np.concatenate((R_points_intersect[index[0]-1, index[1]-1], [select_R[i]]))
            R_points_intersect_index[index[0]-1, index[1]-1] =  np.concatenate([R_points_intersect_index[index[0]-1, index[1]-1], np.array([temp])])
    # Create 2D array of grid cell indices
    # Use digitize() to map red points to grid cell indices
    G_x_indices = np.digitize(select_G[:, 0], grid_x_bounds)
    G_y_indices = np.digitize(select_G[:, 1], grid_y_bounds)
    G_indices = np.column_stack((G_x_indices, G_y_indices))
    # Iterate over the red points and add their coordinates to the corresponding grid cell
    G_points_intersect = np.empty((cols, rows), dtype=object)
    G_points_intersect_index = np.empty((cols, rows), dtype=object)
    for i, index in enumerate(G_indices):
        temp = np.where((G_pos == select_G[i]).all(axis=1))[0][0]
        if G_points_intersect[index[0]-1, index[1]-1] is None:
            G_points_intersect[index[0]-1, index[1]-1] = np.array([select_G[i]])
            G_points_intersect_index[index[0]-1, index[1]-1] = np.array([temp])
        else:
            G_points_intersect[index[0]-1, index[1]-1] = np.concatenate((G_points_intersect[index[0]-1, index[1]-1], [select_G[i]]))
            G_points_intersect_index[index[0]-1, index[1]-1] = np.concatenate([G_points_intersect_index[index[0]-1, index[1]-1], np.array([temp])])
    print(intersection_area)
    # get indices of select_R and select_G in R and G arrays
    select_R_idx = np.where(inside1 == True)[0]
    select_G_idx = np.where(inside2 == True)[0]
    return select_R, select_G, area, R_points_intersect, G_points_intersect, intersection_area, grid_x_bounds, grid_y_bounds, verts,\
           select_R_idx, select_G_idx, R_points_intersect_index, G_points_intersect_index

##############################################################################################################################
####### Select subregions of the area 

def area_select_subregions(R_pos, G_pos, indicator, cell, filedate, path_save, num_rows, num_cols, vertices):
    ## Select certain area of a cell by hand 
    ## Create a scatter plot of the selected points
    fig, ax = plt.subplots()
    ax.scatter(R_pos[:,0], R_pos[:,1], s = 1.5, alpha = 1, color = b2AR_color, label='HALO')
    ax.scatter(G_pos[:,0], G_pos[:,1], s = 1.5, alpha = 1, color = Gs_color, label='SNAP')
    ax.legend()
    ax.set_title(f"{cell}", fontsize = 20)
    ax.set_xlabel(r"x/$\mu$m")
    ax.set_ylabel(r"y/$\mu$m")
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3) 
    cols = num_cols
    rows = num_rows
    # Create a PolygonSelector object and connect it to the callback function)
    if indicator == 'exp':
        poly =  PolygonSelector(ax, oneselect_subregions)
    plt.show()
    if indicator == 'exp':
        R_select, G_select, area, R_points_intersect, G_points_intersect, intersect_area,  \
            grid_x_bounds, grid_y_bounds, verts, \
            select_R_idx, select_G_idx, R_points_intersect_index, G_points_intersect_index \
            = oneselect_subregions(poly.verts, cols, rows, R_pos, G_pos)
        os.makedirs(path_save, exist_ok=True)
        np.savetxt(path_save + f'/second_select_vertex_cell{cell}.txt', np.array(poly.verts))

        vertices_coordinates = np.array(poly.verts)
        np.savetxt(path_save+ f'/Clicked_Vertices_for_Selection{cell}.txt', vertices_coordinates)
    else: 
        R_select, G_select, area, R_points_intersect, G_points_intersect, intersect_area,  grid_x_bounds, grid_y_bounds, verts, \
            select_R_idx, select_G_idx, R_points_intersect_index, G_points_intersect_index \
            = oneselect_subregions(vertices, cols, rows, R_pos, G_pos)
    
    # Plot the selected subregions and the filtered points
    # Plot the filtered points
    fig, ax = plt.subplots()
    ax.scatter(R_select[:,0], R_select[:,1], s = 1.5, alpha = 1, color = b2AR_color , label = 'HALO')
    ax.scatter(G_select[:,0], G_select[:,1], s = 1.5, alpha = 1, color = Gs_color, label = 'SNAP')
    ax.set_xlabel("x/$\mu m$")
    ax.set_ylabel("y/$\mu m$")
    ax.legend()
    # Create grid lines
    for x in grid_x_bounds:
        ax.axvline(x=x, color='blue', alpha=0.5)
    for y in grid_y_bounds:
        ax.axhline(y=y, color='blue', alpha=0.5)  
    # Label nonzero grids with numbers by order
    if np.any(area):
        for i, idx in enumerate(np.argwhere(intersect_area)):
            if len(idx) == 2:
                r, c = idx
                x = (grid_x_bounds[c] + grid_x_bounds[c+1])/2
                y = (grid_y_bounds[r] + grid_y_bounds[r+1])/2
                ax.text(x, y, str(i+1), color='black', ha='center', va='center', fontsize=12)
    # Add the region boundary
    # poly = Polygon(verts, edgecolor='orange', facecolor='none')
    # ax.add_patch(poly)   
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)   
    plt.savefig('selection_subregion.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    if indicator ==  "exp" or indicator == "prev":
        ### Only save files for experimental data
        ### Save basically information 
        info = [len(R_pos), len(G_pos), area, len(select_R_idx), len(select_G_idx)]
        info_names = ['R original number', 'G original number', 'area', 'select_R_number', 'select_G_number']
        with open(path_save + f'/info_cell{cell}.txt', 'w') as f:
           for name, value in zip(info_names, info):
               print(f'{name}: {value}', file=f)
        # Save the selected index as NumPy binary files
        np.savetxt(path_save + f'/R_second_select_idx_cell{cell}.txt', select_R_idx)
        np.savetxt(path_save + f'/G_second_select_idx_cell{cell}.txt', select_G_idx)
        # Convert the object array to a float array
        np.save(path_save + f'/R_second_select_idx_region_cell{cell}', R_points_intersect_index)
        np.save(path_save + f'/G_second_select_idx_region_cell{cell}', G_points_intersect_index)
        np.save(path_save + f'/intersection_area.npy', intersect_area)
        ##### save the second select info
        info2 = [len(R_pos), len(G_pos), intersect_area, len(R_select), len(G_select)]
        info2_names = ['R second number', 'G second number', 'area_new', 'select_R_number second', 'select_G_number second']
        with open(path_save + f'/second_select_info_cell{cell}.txt', 'w') as f:
            for name, value in zip(info2_names, info2):
                print(f'{name}: {value}', file=f)
    return R_select, G_select, area, R_points_intersect, G_points_intersect, intersect_area, R_points_intersect_index, G_points_intersect_index

