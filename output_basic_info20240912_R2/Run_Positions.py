

import pandas as pd
import numpy as np
import os

################################ Load the data 
data_R = pd.read_csv('R_data_PFO20231102_Image3.txt', delimiter='\t')
data_G = pd.read_csv('G_data_PFO20231102_Image3.txt', delimiter='\t')

# Step 2: Read the indices to remove from the text file
with open('R_delete_indice_cluster.txt', 'r') as file:
    indices_to_remove_R = [int(line.strip()) for line in file]

# Step 2: Read the indices to remove from the text file
with open('G_delete_indice_cluster.txt', 'r') as file:
    indices_to_remove_G = [int(line.strip()) for line in file]

# Step 3: Remove the specified indices
deleted_data_R = data_R.drop(indices_to_remove_R)

# Step 3: Remove the specified indices
deleted_data_G = data_G.drop(indices_to_remove_G)

# Step 2: Read the indices to remove from the text file
indices_second_select_R = np.load('Whole_Cell_Region/R_second_select_idx_region_cell20231102_Image3.npy', allow_pickle=True)
# Step 2: Read the indices to remove from the text file
with open('Whole_Cell_Region/R_second_select_idx_cell20231102_Image3.txt', 'r') as file:
    indices_second_long_R = [int(float(line.strip())) for line in file]

data_R_select = np.array(deleted_data_R)[indices_second_long_R]

# Step 2: Read the indices to remove from the text file
indices_second_select_G = np.load('Whole_Cell_Region/G_second_select_idx_region_cell20231102_Image3.npy', allow_pickle=True)
# Step 2: Read the indices to remove from the text file
with open('Whole_Cell_Region/G_second_select_idx_cell20231102_Image3.txt', 'r') as file:
    indices_second_long_G = [int(float(line.strip())) for line in file]

data_G_select = np.array(deleted_data_G)[indices_second_long_G]

# Define the range for i and j
for i in np.arange(0, 1, 1):  # Adjust the range as needed
    for j in np.arange(0, 1, 1):  # Adjust the range as needed
        # Define the file path dynamically based on i and j

        file_path = f'/Users/xcyan/Documents/Harvard_Research/Colocalization_Project_ALL/SSA2/Colocalization_20231216/Preprocessed files Xingchi/Results20231211/Run_Delete_Cluster_Box10/B2AR-Halo__SNAP-CAAX/20231102_B2CAAX_image3_delete/coloca_results20240912_R2/Initial_RG_pair_index_Region/Initial_RG_pair_index_for_Sub_Selection_Region({i}, {j}).txt'
        
        indices_second_select_R_region = indices_second_select_R[i,j]
        indices_second_select_G_region = indices_second_select_G[i,j]
        # Check if the file exists
        if os.path.exists(file_path):
            # Read the R and G indices from the file
            rg_indices = pd.read_csv(file_path, delimiter=',')
            
            # Extract and process R indices
            r_indices = rg_indices.iloc[:, 0].tolist()
            r_indices = [int(round(x)) for x in r_indices]  # Rounding in case of float representations
            
            # Select rows based on R indices (ensuring indices are within bounds)
            pair_R = np.array(indices_second_select_R_region)[r_indices]
            if pair_R.size == 0:
                break
            else: 
                pair_R_list = pair_R.tolist()
                df = np.array(deleted_data_R)
                pos_R =  pd.DataFrame(df[pair_R_list])
                # Save DataFrame to CSV
                file_path2 = f'/Users/xcyan/Documents/Harvard_Research/Colocalization_Project_ALL/SSA2/Colocalization_20231216/Preprocessed files Xingchi/Results20231211/Run_Delete_Cluster_Box10/B2AR-Halo__SNAP-CAAX/20231102_B2CAAX_image3_delete/coloca_results20240912_R2/Pos_Pair_RG/pos_R({i}, {j}).csv'
                pos_R.to_csv(file_path2, sep='\t', index=False)
            
            # Extract and process G indices
            g_indices = rg_indices.iloc[:, 1].tolist()
            g_indices = [int(round(x)) for x in g_indices]  # Rounding in case of float representations
            
            # Select rows based on G indices (ensuring indices are within bounds)
            pair_G = np.array(indices_second_select_G_region)[g_indices]
            
            if pair_G.size == 0:
                break
            else: 
                pair_G_list = pair_G.tolist()
                df = np.array(deleted_data_G)
                pos_G =  pd.DataFrame(df[pair_G_list])
                # Save DataFrame to CSV
                file_path2 = f'/Users/xcyan/Documents/Harvard_Research/Colocalization_Project_ALL/SSA2/Colocalization_20231216/Preprocessed files Xingchi/Results20231211/Run_Delete_Cluster_Box10/B2AR-Halo__SNAP-CAAX/20231102_B2CAAX_image3_delete/coloca_results20240912_R2/Pos_Pair_RG/pos_G({i}, {j}).csv'
                pos_G.to_csv(file_path2, sep='\t', index=False)

            print(f"Pair R for (i, j) = ({i}, {j}):")
            print(pair_R)
            print(f"Pair G for (i, j) = ({i}, {j}):")
            print(pair_G)
        else:
            print(f"File not found: {file_path}")





