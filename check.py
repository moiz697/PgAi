import h5py

# Path to your HDF5 file
hdf5_file_path = 'Save.keras'  # Update with your actual file path

# Open the HDF5 file in read mode
with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    # Navigate to the 'model_weights' group
    model_weights_group = hdf5_file['model_weights']

    # Optionally, you can print more details about each layer's weights
    for layer_name in model_weights_group.keys():
        layer_group = model_weights_group[layer_name]
        
        # Use encode('utf-8', 'ignore').decode('utf-8') to handle decoding errors
        layer_name_str = layer_name.encode('utf-8', 'ignore').decode('utf-8')

        print(f"\nLayer: {layer_name_str}")
        print("Weights:")
        
        # Iterate over the items using .items() to get both key and value
        for weight_name, weight_data in layer_group.items():
            # Use encode('utf-8', 'ignore').decode('utf-8') to handle decoding errors
            weight_name_str = weight_name.encode('utf-8', 'ignore').decode('utf-8')
            
            # Access the array using [()] if it's a dataset (not a group)
            if isinstance(weight_data, h5py.Dataset):
                try:
                    weight_array = weight_data[()]
                    print(f"  {weight_name_str}: {weight_array.shape}")
                except Exception as e:
                    print(f"  {weight_name_str}: Error accessing array - {str(e)}")
