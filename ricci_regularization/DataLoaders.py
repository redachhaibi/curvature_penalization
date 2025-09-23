import json
from sklearn import datasets
import torch, torchvision
from torch.utils.data import Subset
from torchvision import datasets, transforms
import sklearn
import ricci_regularization
from tqdm.notebook import tqdm


def get_dataloaders(dataset_config: dict, data_loader_config: dict, dtype: str, datasets_root = None):
    """
    Creates and returns training and testing data loaders for different datasets.

    Supported datasets:
      - MNIST (full dataset)
      - MNIST_subset (random subset with user-specified labels and samples per label)
      - Synthetic (custom synthetic dataset from ricci_regularization.SyntheticDataset)
      - Swissroll (synthetic dataset generated using sklearn's make_swiss_roll)

    Args:
        dataset_config (dict): Configuration dictionary for the dataset, may include:
            - "name": dataset name ("MNIST", "MNIST_subset", "Synthetic", "Swissroll")
            - dataset-specific parameters (e.g., selected_labels, num_points_per_label, n, k, d, etc.)
        data_loader_config (dict): Configuration dictionary for the dataloaders, must include:
            - "batch_size": batch size for the dataloaders
            - "split_ratio": fraction of data to use for testing
            - "random_seed": random seed for reproducibility
            - "random_shuffling": whether to shuffle training data
        dtype (str): Torch dtype for tensors (e.g., torch.float32, torch.float64).
        datasets_root (str, optional): Root directory for storing/loading datasets. 
                                       Defaults to "../../datasets/".

    Returns:
        dict: A dictionary containing:
            - "train_loader": DataLoader for the training set
            - "test_loader": DataLoader for the testing set
            - "test_dataset": The full test dataset (or split, depending on dataset type)
            - "test_dataset_partial": (only for MNIST, MNIST_subset) partial test set
    """
    if datasets_root == None:
        datasets_root = '../../datasets/'  # Root directory for datasets

    # Load dataset based on the name provided in dataset_config
    torch.manual_seed(data_loader_config["random_seed"])  # Set the random seed for reproducibility

    # Define the transformation: Convert to tensor and change the type to float64
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # Converts the PIL image to torch.Tensor (default: torch.float32)
            torchvision.transforms.Lambda(lambda x: x.to(dtype))  # Convert the tensor to torch.float64
        ])
    if dataset_config["name"] == "MNIST":
        # Load the MNIST dataset
        dataset = datasets.MNIST(root=datasets_root, train=True, transform=transform, download=True)
        test_dataset  = datasets.MNIST(root=datasets_root, train=False, transform=transform, download=False)

    elif dataset_config["name"] == "MNIST_subset":
        # Load the full MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root=datasets_root, train=True, transform=transform, download=True)
        test_dataset  = torchvision.datasets.MNIST(root=datasets_root, train=False, transform=transform, download=False)
        # Selecting specific labels in train dataset
        num_points_per_label = dataset_config["num_points_per_label"]
        indices_random = [] # list to store randomely selected indices 
        selected_labels = dataset_config["selected_labels"]  # Get the list of labels to select
        for label in selected_labels:
            mask = torch.isin(train_dataset.targets, torch.tensor(label))
            indices = torch.where(mask)[0]
            indices_random.append(indices[torch.randperm(len(indices))][:num_points_per_label])
        indices_random = torch.cat(indices_random)
        # permuting concateneted indices
        indices_random = indices_random[torch.randperm(len(indices_random))]
        dataset = torch.utils.data.Subset(train_dataset, indices_random)
        sizeof_dataset = len(dataset)
        # Split the dataset into training and testing sets based on split_ratio
        train_data, test_data = torch.utils.data.random_split(dataset, [sizeof_dataset - int(sizeof_dataset * data_loader_config["split_ratio"]), int(sizeof_dataset * data_loader_config["split_ratio"])])

    elif dataset_config["name"] == "Synthetic":
        # Generate a synthetic dataset using specified parameters
        my_dataset = ricci_regularization.SyntheticDataset(
            k=dataset_config["k"], 
            n=dataset_config["n"],
            d=dataset_config["d"], 
            D=dataset_config["D"], 
            shift_class=dataset_config["shift_class"],
            interclass_variance=dataset_config["interclass_variance"], 
            variance_of_classes=dataset_config["variance_of_classes"]
        )
        dataset = my_dataset.create  # Create the synthetic dataset
    elif dataset_config["name"] == "Swissroll":
        # Generate the Swissroll dataset
        train_dataset = sklearn.datasets.make_swiss_roll(n_samples=dataset_config["n"], 
            noise=dataset_config["swissroll_noise"],random_state = data_loader_config["random_seed"])
        
        sr_points = torch.from_numpy(train_dataset[0]).to(torch.float32)  # Convert points to tensor
        sr_colors = torch.from_numpy(train_dataset[1]).to(torch.float32)  # Convert colors to tensor
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(sr_points, sr_colors)  # Create a TensorDataset from points and colors
    
    m = len(dataset)  # Get the length of the training dataset
    # Split the dataset into training and testing sets based on split_ratio
    train_data, test_data = torch.utils.data.random_split(dataset, [m - int(m * data_loader_config["split_ratio"]), int(m * data_loader_config["split_ratio"])])

    # Create data loaders for training and testing sets
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=data_loader_config["batch_size"])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=data_loader_config["batch_size"], shuffle=data_loader_config["random_shuffling"])
    
    # Return a dictionary containing the training and testing data loaders
    loaders = {
        "train_loader": train_loader,
        "test_loader": test_loader
    }
    if dataset_config["name"] in ["MNIST","MNIST_subset"]:
        loaders["test_dataset"] = test_dataset
        loaders["test_dataset_partial"] = test_data
    else:
        loaders["test_dataset"] = test_data
    return loaders


# This function returns the initialized and loaded model  and the path to the model weights.
# It is used to build reports after training when one whants to load the weights of a pre6trained AE.
def get_tuned_nn(config: dict, additional_path = '', verbose = True):
    """
    Initialize and load a pre-trained neural network model with weights.

    This function constructs a neural network architecture (currently supports 
    TorusAE and TorusConvAE), loads its pre-trained weights from disk, 
    moves it to GPU if available, and sets it to evaluation mode.

    Args:
        config (dict): Configuration dictionary with the following keys:
            - "architecture": dict containing:
                - "latent_dim": latent space dimension
                - "input_dim": input dimension
                - "type" (optional): architecture type, either "TorusAE" or "TorusConvAE".
                                    Defaults to "TorusAE" if not specified.
            - "experiment": dict containing:
                - "name": name of the experiment (used to locate saved weights)
        additional_path (str, optional): Additional prefix path for locating the weights.
                                         Defaults to ''.
        verbose (bool, optional): If True, prints the chosen architecture. Defaults to True.

    Returns:
        tuple:
            - torch.nn.Module: The initialized model with loaded weights (in eval mode).
            - str: Path to the loaded model weights.
    """
    latent_dim = config["architecture"]["latent_dim"]
    input_dim  = config["architecture"]["input_dim"]

    try:
        # Attempt to retrieve the architecture type from the configuration dictionary
        architecture_type = config["architecture"]["type"]
    except KeyError:
        # Default to "TorusAE" if the architecture type is not specified
        architecture_type = "TorusAE"
    if verbose == True:
        print("Chosen architecture:",architecture_type)
    # Initialize the neural network based on the specified architecture type
    if architecture_type == "TorusAE":
        torus_ae = ricci_regularization.Architectures.TorusAE(x_dim=input_dim, h_dim1=512, h_dim2=256, z_dim=latent_dim)
    elif architecture_type == "TorusConvAE":
        torus_ae = ricci_regularization.Architectures.TorusConvAE(x_dim=input_dim, h_dim1=512, h_dim2=256, z_dim=latent_dim, pixels=28)
    
    # If a GPU is available, move the model to the GPU
    if torch.cuda.is_available():
        torus_ae.cuda()

    # Construct the path to the saved model weights using the experiment name from the configuration
    PATH_ae_wights = additional_path + "../experiments/" + config["experiment"]["name"] + "/ae_weights.pt"
    
    # Load the saved model weights from the specified path
    torus_ae.load_state_dict(torch.load(PATH_ae_wights))
    
    # Set the model to evaluation mode
    torus_ae.eval()
    
    return torus_ae, PATH_ae_wights