import torch, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal

def make_grid(numsteps, 
              xcenter = 0.0, ycenter = 0.0,
              xsize =  3.0, ysize = 3.0,
              xshift = 0.0, yshift = 0.0):
    """
    Generate a 2D grid of points within a specified range.
    
    Parameters:
    - numsteps: Number of steps (grid points) along each axis.
    - xcenter, ycenter: Center coordinates of the grid.
    - xsize, ysize: Total width and height of the grid.
    - xshift, yshift: Optional shifts to move the grid in x or y direction.

    Returns:
    - tgrid: A tensor containing the grid points in (x, y) coordinate pairs.
    """
    
    xs = torch.linspace(xcenter - 0.5*xsize, xcenter + 0.5*xsize, steps = numsteps) + xshift
    ys = torch.linspace(ycenter - 0.5*ysize, ycenter + 0.5*ysize, steps = numsteps) + yshift
    
    # True grid starts from the bottom-left corner. x is the first to increase.
    tgrid = torch.cartesian_prod(ys, xs)
    tgrid = tgrid.roll(1, 1)
    return tgrid

# plot recostructions for mnist dataset
def plot_ae_outputs(test_dataset, encoder, decoder, n=10, D=784):
    """
    Plot original and reconstructed images using an autoencoder.

    Parameters:
    - test_dataset: A dataset containing test images and their labels.
    - encoder: Trained encoder model.
    - decoder: Trained decoder model.
    - n: Number of different digits to visualize (default: 10).
    - D: Flattened image dimension (default: 784 for 28x28 images).

    Returns:
    - A matplotlib plot displaying original images (top row) 
      and their corresponding reconstructed images (bottom row).
    """

    plt.figure(figsize=(16 * n / 10, 4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0)

        with torch.no_grad():
            rec_img = decoder(encoder(img.reshape(1, D))).reshape(1, 28, 28)

        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title('Original images')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title('Reconstructed images')

    plt.show()

def plot_ae_outputs_selected(test_dataset, encoder, decoder, targets=None, selected_labels=None, D=784, dpi=400):
    """
    Plot original and reconstructed images for selected labels using an autoencoder.

    Parameters:
    - test_dataset: A dataset containing test images and their labels.
    - encoder: Trained encoder model.
    - decoder: Trained decoder model.
    - targets: Optional, array of target labels corresponding to the dataset.
    - selected_labels: List of specific labels to visualize. If None, defaults to 10 labels.
    - D: Flattened image dimension (default: 784 for 28x28 images).
    - dpi: Resolution of the plot.

    Returns:
    - A matplotlib subplot displaying selected original images (top row) 
      and their corresponding reconstructed images (bottom row).
    """

    if selected_labels is None:
        n = 10  # Default number of images
    else:
        n = len(selected_labels)

    plt.figure(figsize=(6 * n / 10, 1.5), dpi=dpi)

    if targets is None:
        targets = test_dataset.targets.numpy()

    t_idx = {i: np.where(targets == i)[0][0] for i in selected_labels}

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[selected_labels[i]]][0].unsqueeze(0)

        with torch.no_grad():
            rec_img = decoder(encoder(img.reshape(1, D))).reshape(1, 28, 28)

        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title('Original images', fontsize=6)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title('Reconstructed images', fontsize=6)

    return ax

def discrete_cmap(N, base_cmap=None, bright_colors = False):
    """
    Create an N-bin discrete colormap from the specified input map
    borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    if (N==2) and (bright_colors ==True):
        color_list = base(np.array([0.3,0.85]))

    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# this function is not used but can be used for fast and direct loss plotting from dict
def plotfromdict(dict_of_losses):
    """
    Plots multiple loss curves from a dictionary of loss lists.

    Parameters:
        dict_of_losses (dict): A dictionary where keys are loss names (str) 
                               and values are lists of loss values.

    Returns:
        fig, axes: The created matplotlib figure and axes.
    """

    number_of_plots = len(dict_of_losses)  # Determine the number of subplots needed
    fig, axes = plt.subplots(nrows=number_of_plots, ncols=1, figsize=(6, number_of_plots * 6))  
    # Create subplots with dynamic height
    i = 0  
    color_iterable = iter(mcolors.TABLEAU_COLORS)  # Get an iterable for colors
    for name, loss_list in dict_of_losses.items():  
        if number_of_plots == 1:
            axes = [axes]  # Ensure axes is always iterable when there's only one plot

        try:
            newcolor = next(color_iterable)  # Assign a new color to the plot
        except StopIteration:
            # Reset the color iterator if it runs out of colors
            color_iterable = iter(mcolors.TABLEAU_COLORS)
            newcolor = next(color_iterable)

        axes[i].semilogy(loss_list, color=newcolor)  # Plot loss on a semi-log scale
        axes[i].set_ylabel(name)  # Label the y-axis with the loss name
        axes[i].set_xlabel('Batches')  # Label the x-axis as 'Batches'
        i += 1  
    plt.show()  
    return fig, axes  

def plotsmart(dictplots):
    """
    Plots multiple datasets from a dictionary, using semi-log scale.

    Parameters:
        dictplots (dict): A dictionary where each key is a plot name, 
                          and each value is another dictionary with:
                          - "data": A dictionary of curves (keys are legends, values are lists of data points).
                          - "yname_latex": Label for the y-axis.

    Returns:
        fig, axes: The created matplotlib figure and axes.
    """
    number_of_plots = len(dictplots)  # Number of subplots required
    fig, axes = plt.subplots(nrows=number_of_plots, ncols=1, figsize=(4, number_of_plots * 4))  # Create subplots with dynamic height
    i = 0  
    color_iterable = iter(mcolors.TABLEAU_COLORS)  # Iterable for color selection
    for plot_name, plot_info in dictplots.items():  
        if number_of_plots == 1:
            axes = [axes]  # Ensure `axes` is always iterable when there's only one plot
        for legend, curve in plot_info["data"].items():  
            try:
                newcolor = next(color_iterable)  # Assign a new color to each curve
            except StopIteration:
                color_iterable = iter(mcolors.TABLEAU_COLORS)  # Reset color iterator if colors run out
                newcolor = next(color_iterable)
            linestyle = 'dashed' if legend in {"max", "min"} else 'solid'  # Determine line style based on legend type
            axes[i].semilogy(curve, color=newcolor, label=legend, ls=linestyle)  # Plot curve with semi-log scale
        axes[i].set_ylabel(plot_info["yname_latex"])  # Set y-axis label
        axes[i].set_xlabel('Batches')  # Set x-axis label
        axes[i].legend(loc="lower left")  # Add legend to the plot
        i += 1  
    plt.show()  
    return fig, axes  # Return the figure and axes for further modifications if needed

def translate_dict(dict_losses_to_plot, eps = 0):
    """
    Translates a dictionary of loss values into a format suitable for plotting.
    
    Parameters:
        dict_losses_to_plot (dict): Dictionary containing the loss values for various metrics.
        eps (float): A parameter used in some of the loss plots (default is 0).

    Returns:
        dictplots (dict): A dictionary structured for plotting, with each plot containing
                          its respective loss data and labels.
    """
    # Construct the dictionary with various plot configurations
    dictplots = {
        "plot1": {
            "yname_latex": "MSE",  # Label for y-axis
            "data": {
                "MSE": dict_losses_to_plot["MSE"]  # Loss data for MSE
            }
        },
        "plot2": {
            "yname_latex": "$\mathcal{L}_\mathrm{equi}$",  # Label for Equidistribution loss
            "data": {
                "$\widehat\mathcal{L}_\mathrm{equi}$": dict_losses_to_plot["Equidistribution"]  # Data for Equidistribution loss
            }
        },
        "plot3": {
            "yname_latex": "$\mathcal{L}_\mathrm{contractive}$",  # Label for Contractive loss
            "data": {"$\widehat\mathcal{L}_\mathrm{contractive}$": dict_losses_to_plot.get("Contractive")}  # Data for Contractive loss
        },
        "plot4": {
            "yname_latex": "$\mathcal{L}_\mathrm{curv}$",  # Label for Curvature loss
            "data": {"$\widehat\mathcal{L}_\mathrm{curv}$": dict_losses_to_plot.get("Curvature")}  # Data for Curvature loss
        },
        "plot5": {
            "yname_latex": "$\det(g)$",  # Label for determinant of 'g'
            "data": {k: dict_losses_to_plot.get(k) for k in ["g_det_mean", "g_det_max", "g_det_min"]}  # Data for determinant of 'g'
        },
        "plot6": {
            "yname_latex": rf"$\|g_{{reg}}^{{-1}}\|_F, \ \varepsilon = {eps}$",  # Label for regularized inverse norm
            "data": {k: dict_losses_to_plot.get(k) for k in ["g_inv_norm_mean", "g_inv_norm_max"]}  # Data for g inverse norm
        },
        "plot7": {
            "yname_latex": r"$\|\nabla \Psi \|^2_F = \mathrm{tr} (g) $",  # Label for decoder Jacobian norm
            "data": {k: dict_losses_to_plot.get(k) for k in ["decoder_jac_norm_mean", "decoder_jac_norm_max"]}  # Data for decoder Jacobian norm
        },
        "plot8": {
            "yname_latex": r"$\|\nabla \Phi \|^2_F $",  # Label for encoder Jacobian norm
            "data": {k: dict_losses_to_plot.get(k) for k in ["encoder_jac_norm_mean", "encoder_jac_norm_max"]}  # Data for encoder Jacobian norm
        },
        "plot9": {
            "yname_latex": "$R^2$",  # Label for R^2
            "data": {k: dict_losses_to_plot.get(k) for k in ["curv_squared_mean", "curv_squared_max"]}  # Data for R^2
        }
    }

    # Merge and filter out empty plots (i.e., plots where no data exists)
    keys_to_remove = []  # List to keep track of keys to remove

    # Iterate over dictplots and check if any "data" contains None values (i.e., missing data)
    for key, values in dictplots.items():
        if None in values["data"].values():  # Check if there are any None values in the "data"
            keys_to_remove.append(key)  # Mark this key for removal

    # Remove the keys with missing data
    for key in keys_to_remove:
        dictplots.pop(key, None)  # Remove the key if it exists in dictplots

    return dictplots  # Return the filtered dictionary ready for plotting

def PlotSmartConvolve(dictplots, test_dictplots=None, plot_test_losses=True,
        numwindows1=50, numwindows2=200):
    """
    Plot training and testing curves with convolution smoothing applied.
    
    Parameters:
        dictplots (dict): Dictionary of training data to plot.
        test_dictplots (dict): Dictionary of test data to plot.
        plot_test_losses (bool): Flag indicating whether to plot test losses.
        numwindows1 (int): Size of the first convolution window.
        numwindows2 (int): Size of the second convolution window.

    Returns:
        fig, axes: The figure and axes of the created plot.
    """
    if plot_test_losses == True and test_dictplots == None:
        print("Set test losses dictionary to print!")
        return
    
    number_of_plots = len(dictplots)  # Number of plots to create

    # Create a figure with multiple subplots (3 columns for each plot)
    fig, axes = plt.subplots(nrows=number_of_plots, ncols=3, figsize=(4*3, number_of_plots*4))
    
    # Define convolution windows of different sizes (Hann windows)
    win = [signal.windows.hann(1), signal.windows.hann(numwindows1), signal.windows.hann(numwindows2)]  

    i = 0
    color_iterable = iter(mcolors.TABLEAU_COLORS)  # Color iterator for plotting
    for plot_name, plot_info in dictplots.items():
        for legend, curve in plot_info["data"].items():
            try:
                newcolor = next(color_iterable)  # Get next color
            except StopIteration:
                color_iterable = iter(mcolors.TABLEAU_COLORS)
                newcolor = next(color_iterable)

            linestyle = 'dashed' if legend == "max" else 'solid'  # Set line style based on legend

            # Apply convolution and plot for each of the 3 windows
            for j in range(3):
                axes[i, j].semilogy(
                    signal.convolve(curve, win[j], mode='valid') / sum(win[j]), 
                    color=newcolor, label="Train", ls=linestyle
                )
                axes[i, j].set_xlabel('Batches')

        axes[i, 0].set_ylabel(plot_info["yname_latex"])
        i += 1

    if plot_test_losses == True:
        i = 0
        # Plot test losses
        for plot_name, plot_info in test_dictplots.items():
            for legend, test_curve in plot_info["data"].items():
                try:
                    newcolor = next(color_iterable)
                except StopIteration:
                    color_iterable = iter(mcolors.TABLEAU_COLORS)
                    newcolor = next(color_iterable)
                
                pace_ratio = len(curve) / len(test_curve)
                test_curve_interpolated = np.interp(
                    np.arange(len(curve)), np.arange(len(test_curve)) * pace_ratio, test_curve
                )  # Interpolate test curve to match train curve length

                # Apply convolution and plot for each of the 3 windows
                for j in range(3):
                    axes[i, j].semilogy(
                        signal.convolve(test_curve_interpolated, win[j], mode='valid') / sum(win[j]), 
                        color=newcolor, label="Test", ls=linestyle
                    )
            axes[i, 0].legend(loc="lower left")
            i += 1
    
    plt.show()  # Display the plot
    return fig, axes  # Return the figure and axes for further modifications

def point_plot(encoder, data_loader, batch_idx, config,
        show_title=True, colormap='jet', normalize_to_unit_square=False, 
        s=40, draw_grid=False, figsize=(9, 9)):
    """
    Plots the latent space representation of data points using an encoder.
    Takes the dataloader (test or train) and plots the latent space.
    
    Parameters:
        encoder (nn.Module): The model that encodes the data.
        data_loader (DataLoader): DataLoader providing the data.
        batch_idx (int): Current batch index (used in title).
        config (dict): Configuration dictionary (dataset, input dimensions).
        show_title (bool): Flag to show the plot title.
        colormap (str): Colormap to use for coloring the points.
        normalize_to_unit_square (bool): Flag to normalize points to the unit square.
        s (int): Size of the scatter plot points.
        draw_grid (bool): Flag to enable/disable grid.
        figsize (tuple): Size of the figure.

    Returns:
        fig: The created figure.
    """
    # Extract all data and labels from the data_loader
    data_tensor_list = []
    labels_list = []
    for batch_data_labels in data_loader:
        batch_data, batch_labels = batch_data_labels
        data_tensor_list.append(batch_data)
        labels_list.append(batch_labels)
    data_tensor = torch.cat(data_tensor_list)
    labels = torch.cat(labels_list)
    
    D = config["architecture"]["input_dim"]
    dataset_name = config["dataset"]["name"]

    # Perform encoding
    with torch.no_grad():
        data_tensor = data_tensor.view(-1, D)  # Flatten the data
        encoded_data = encoder(data_tensor).cpu()

    encoded_data_to_plot = encoded_data.numpy()
    labels = labels.numpy()
    selected_labels = np.unique(labels)

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if needed
    if normalize_to_unit_square:
        encoded_data = encoded_data / torch.pi
    else:
        ax.set_ylim(-math.pi, math.pi)
        ax.set_xlim(-math.pi, math.pi)
        ax.set_yticks([-3., -2., -1., 0., 1., 2., 3.])
        ax.set_xticks([-3., -2., -1., 0., 1., 2., 3.])
    
    # Create scatter plot with appropriate color based on dataset
    if dataset_name == "Swissroll":
        sc = ax.scatter(encoded_data_to_plot[:, 0], encoded_data_to_plot[:, 1], s=s, c=labels, alpha=1.0, 
                        marker='o', edgecolor='none', cmap=colormap)
    elif dataset_name in ["MNIST_subset", "MNIST"]:
        k = len(selected_labels)
        norm = plt.Normalize(vmin=min(labels), vmax=max(labels))
        sc = ax.scatter(encoded_data_to_plot[:, 0], encoded_data_to_plot[:, 1], s=s, c=labels, alpha=1.0, 
                        marker='o', edgecolor='none', cmap=colormap)
        
        cmap = plt.get_cmap(colormap)
        discrete_cmap = mcolors.ListedColormap([cmap(norm(label)) for label in selected_labels])
        sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
        sm.set_array([])
        tick_positions = np.linspace(selected_labels.min(), selected_labels.max(), k)
        cbar = fig.colorbar(sm, ax=ax, ticks=tick_positions, shrink=0.7, spacing="uniform")
        cbar.set_label('Cluster Label')
        cbar.set_ticklabels(selected_labels)  # Show only unique labels

    # Add title if required
    if show_title:
        ax.set_title(f'Latent space for test data in AE at batch {batch_idx}')
    
    # Enable grid if required
    ax.grid(draw_grid)

    # Adjust layout to prevent cutting off of elements
    fig.tight_layout()

    # Return the figure
    return fig

def point_plot_fast(encoded_points, labels, batch_idx, config, show_title=True, colormap='jet', 
               s=40, draw_grid=False, figsize=(9, 9), Saving_path=None):
    """
    A faster version of the point_plot function designed for plotting during training.
    Takes the encoded points.

    Parameters:
        encoded_points (ndarray): Encoded data points to plot.
        labels (ndarray): Corresponding labels for the data points.
        batch_idx (int): Current batch index (used in title).
        config (dict): Configuration dictionary (dataset, labels).
        show_title (bool): Flag to show the plot title.
        colormap (str): Colormap for the points.
        s (int): Size of the scatter plot points.
        draw_grid (bool): Flag to enable/disable grid.
        figsize (tuple): Size of the figure.
        Saving_path (str): Path to save the figure.

    Returns:
        fig: The created figure.
    """
    dataset_name = config["dataset"]["name"]

    # Create figure and axes for the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot for Swissroll or MNIST-like datasets
    if dataset_name == "Swissroll":
        sc = ax.scatter(encoded_points[:, 0], encoded_points[:, 1], s=s, c=labels, alpha=1.0, marker='o', 
                        edgecolor='none', cmap=colormap)
    elif dataset_name in ["MNIST_subset", "MNIST", "Synthetic"]:
        if config["dataset"]["name"] == "MNIST":
            selected_labels = torch.arange(10)
        elif config["dataset"]["name"] == "Synthetic":
            selected_labels = torch.arange(config["dataset"]["k"])
        else:
            selected_labels = torch.tensor(config["dataset"]["selected_labels"])
        
        mask = torch.isin(labels, torch.tensor(selected_labels))
        filtered_data = encoded_points[mask]
        filtered_labels = labels[mask]

        sc = ax.scatter(filtered_data[:, 0], filtered_data[:, 1], s=s, c=filtered_labels, alpha=1.0, 
                        marker='o', edgecolor='none', cmap=colormap)
        
        cmap = plt.get_cmap(colormap)
        norm = plt.Normalize(vmin=torch.min(selected_labels).item(), vmax=torch.max(selected_labels).item())
        discrete_cmap = mcolors.ListedColormap([cmap(norm(label)) for label in selected_labels.tolist()])
        sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
        sm.set_array([])
        tick_positions = np.linspace(selected_labels.min(), selected_labels.max(), len(selected_labels))
        cbar = fig.colorbar(sm, ax=ax, ticks=tick_positions, shrink=0.7, spacing="uniform")
        cbar.set_label('Cluster Label')
        cbar.set_ticklabels(selected_labels.tolist())

    # Add title if required
    if show_title:
        ax.set_title(f'Latent space for test data in AE at batch {batch_idx}')
    
    # Enable grid if required
    ax.grid(draw_grid)

    # Adjust layout to prevent clipping
    fig.tight_layout()

    if Saving_path is not None:
        fig.savefig(Saving_path + f"/latent_space_at_batch_{batch_idx}.pdf", bbox_inches='tight', format="pdf")

    # Return the figure
    return fig


def plot_2d_encoding(encoded_points, color_labels, cmap, opacity, 
                     save_plot=False, size_of_points = 40, verbose=True, 
                     Saving_file_name=None, Plot_title= None):
    """
    Plots a 2D scatter of encoded points with optional coloring, opacity, and saving.

    This function is typically used to visualize the 2D latent space of an autoencoder or other 
    dimensionality-reduced data. Each point can be colored according to its label or another variable.

    Args:
        encoded_points (torch.Tensor or np.ndarray): Tensor/array of shape [num_points, 2] containing 2D coordinates.
        color_labels (torch.Tensor or np.ndarray): Values to color each point, same length as encoded_points.
        cmap (str or matplotlib colormap): Colormap used for coloring points.
        opacity (float): Alpha value for points (0 transparent - 1 opaque).
        save_plot (bool, optional): If True, saves the plot to file. Defaults to False.
        size_of_points (int, optional): Size of scatter points. Defaults to 40.
        verbose (bool, optional): If True, displays the plot. If False, closes it. Defaults to True.
        Saving_file_name (str, optional): File path to save the plot if save_plot=True. Defaults to None.
        Plot_title (str, optional): Title of the plot. Defaults to None.

    Notes:
        - If save_plot=True and Saving_file_name is not provided, a warning is printed and the plot is not saved.
        - Designed for 2D embeddings. For higher dimensions, reduce to 2D before plotting.
    """
    plt.figure(figsize=(9,9),dpi=400)
    plt.rcParams.update({'font.size': 20}) # makes all fonts on the plot be 20
    plt.scatter( encoded_points[:,0], encoded_points[:,1], 
                c=color_labels, s= size_of_points,
                alpha=opacity, 
                cmap=cmap,
                marker='o',
                edgecolors=None )
    if save_plot == True:
        if Saving_file_name == None:
            print("file cannot be saved, please set file name in Saving_file_name")
        else:
            plt.savefig(Saving_file_name,bbox_inches='tight',format='pdf')
    if Plot_title != None:
        plt.title(Plot_title)
    if verbose == True:
        plt.show()
    else:
        plt.close()

def plot_histogram(data,title,Path_pictures,
                        filename="histogram.pdf",
                        violent_saving=False, 
                        percentile_bounds=(1, 99),
                        bins=200,
                        verbose=True):
    """
    Plot and save histogram of data, excluding outliers.

    Parameters
    ----------
    data : array-like
        Input data values.
    title : str
        Title of the plot.
    Path_pictures : str
        Directory path where the plot will be saved if violent_saving=True.
    filename : str, default="histogram.pdf"
        File name for saving the plot (only used if violent_saving=True).
    violent_saving : bool, default=False
        Whether to save the figure as a PDF.
    percentile_bounds : tuple, default=(1, 99)
        Lower and upper percentile bounds for filtering outliers.
    bins : int, default=200
        Number of histogram bins.
    verbose: bool, default=True
        Whether to show the plot.
    """

    # Define outlier limits using percentiles
    lower_bound, upper_bound = np.percentile(data, percentile_bounds)

    # Filter out outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    # Compute statistics
    mean_value = torch.mean(data)
    std_value = torch.std(data)

    # Plot histogram
    plt.figure(figsize=(5, 3), dpi=200)
    plt.rcParams.update({'font.size': 10})
    plt.title(title)

    stats_text = (f"Mean: {mean_value:.8f}\n"
                  f"Std: {std_value:.8f}")

    plt.hist(filtered_data, bins=bins, density=True)

    # Get updated axis limits *after* plotting
    ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # Place stats text dynamically in top-right corner
    ax.text(xlim[1] * 0.98, ylim[1] * 0.98, stats_text,
            ha='right', va='top', fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "black"})

    # Save figure if required
    if violent_saving:
        plt.savefig(f'{Path_pictures}/{filename}', bbox_inches='tight', format='pdf')
    if verbose == True:
        plt.show()
    else:
        plt.close()