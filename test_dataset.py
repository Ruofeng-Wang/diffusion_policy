import zarr
import numpy as np
import matplotlib.pyplot as plt

def load_and_plot(zarr_path, dataset_path):
    # Load Zarr data
    store = zarr.open(zarr_path, mode='r')
    actions = store[dataset_path]  # For example, 'data/action'

    actions = actions[-500:]

    # Check if actions is a single dimension data or multidimensional
    if actions.ndim == 1:
        plt.plot(actions)
        plt.title('Action Data Over Time')
        plt.xlabel('Time')
        plt.ylabel('Action Value')
    elif actions.ndim > 1:
        # Plotting only the first component if multidimensional, you can adjust as needed
        plt.plot(actions[:, 2])  # Adjust the index if you want to plot different components
        plt.title('First Component of Action Data Over Time')
        plt.xlabel('Time')
        plt.ylabel('First Component Value')


if __name__ == '__main__':
    zarr_path = 'recorded_data_combined_12-21-12.zarr'  # Replace with the path to your Zarr file
    dataset_path = 'data/action'  # Adjust the dataset path inside the Zarr file if needed
    load_and_plot(zarr_path, dataset_path)
    load_and_plot('recorded_data_combined_19-10-39.zarr', dataset_path)

    # Show plot
    plt.show()