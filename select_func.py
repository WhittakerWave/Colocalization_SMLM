

###########################################################################################
#### function used to select a certain region
from package_func import *

##### Color for plotting 
b2AR_color = tuple(c/255 for c in [0, 176, 80])
Gs_color = tuple(c/255 for c in [255, 0, 255])
complex_color = tuple(c/255 for c in [63, 83, 135])
#################################################################################################
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

############################################################################################################
#### function to selection an area of interest and return the index (compared with input) in the selected region
def area_selection(R_pos, G_pos, Cell, filedate, path_save, selection_number):
    ## Select certain area of a cell by hand 
    ## Create a scatter plot of the selected points
    fig, ax = plt.subplots()
    ax.scatter(R_pos[:,0], R_pos[:,1], s = 0.5, alpha = 0.85, label='Receptor', color = b2AR_color)
    ax.scatter(G_pos[:,0], G_pos[:,1], s = 0.5, alpha = 0.85, label='G Protein', color = Gs_color)
    ax.legend()
    ax.set_title(f"{Cell}", fontsize = 20)
    ax.set_xlabel(r"x/$\mu$m")
    ax.set_ylabel(r"y/$\mu$m")
    useLargeSize(plt, ax, marker_lines = None, fontsize = 'xx-large',fontproperties=None, LW=2.3)  
    # Create a PolygonSelector object and connect it to the callback function
    poly = PolygonSelector(ax, onselect)
    plt.show()
    _, _, select_R_idx, select_G_idx, area = onselect(poly.verts,  R_pos, G_pos)
    density_R_select = len(select_R_idx)/area
    density_G_select = len(select_G_idx)/area
    info = [len(R_pos), len(G_pos), area, len(select_R_idx), len(select_G_idx), density_R_select, density_G_select]
    info_names = ['R number before selection', 'G number before selection', 'area', 'select_R_number', 'select_G_number', 'Density R after selection', 'Density G after selection']
    folder_name = f'Area Selection Number{selection_number}'
    folder_path = os.path.join(path_save, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    np.savetxt(folder_path + f'/Poly_vertices_for_Selection{selection_number}.txt', poly.verts)
    with open(folder_path + f'/info_cell{Cell}_for_Selection{selection_number}.txt', 'w') as f:
        for name, value in zip(info_names, info):
            print(f'{name}: {value}', file=f)
    # Save the selected index as NumPy binary files
    np.savetxt(folder_path  +f'/R_select_idx_cell{Cell}_for_Selection{selection_number}.txt', select_R_idx)
    np.savetxt(folder_path  +f'/G_select_idx_cell{Cell}_for_Selection{selection_number}.txt', select_G_idx)
    print(area)
    return select_R_idx, select_G_idx, area

############################################################################################################
##### function to save the files to analyze 

def save_files_analyze(R_pos_analyze, G_pos_analyze, R_prec_analyze, G_prec_analyze, R_psf_analyze, G_psf_analyze, R_frame_analyze, G_frame_analyze, path, cell, selection_number):
    R_data_analyze  = np.column_stack((R_frame_analyze, R_pos_analyze[:, 0], R_pos_analyze[:, 1], R_prec_analyze, R_psf_analyze))
    G_data_analyze  = np.column_stack((G_frame_analyze, G_pos_analyze[:, 0], G_pos_analyze[:, 1], G_prec_analyze, G_psf_analyze))
    # Save the data to the text file
    # Define the column names
    column_names = ['Frame', 'pos_x/um', 'pos_y/um', 'precision/nm', 'psf/nm']
    # Insert the column names as the first row
    R_data_analyze = np.vstack((column_names, R_data_analyze))
    G_data_analyze = np.vstack((column_names, G_data_analyze))
    folder_name = f'Area Selection Number{selection_number}'
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    np.savetxt(folder_path + f'/R_data_analyze_{cell}_for_Selection{selection_number}.txt' , R_data_analyze, fmt='%s', delimiter='\t')
    np.savetxt(folder_path + f'/G_data_analyze_{cell}_for_Selection{selection_number}.txt' , G_data_analyze, fmt='%s', delimiter='\t')

