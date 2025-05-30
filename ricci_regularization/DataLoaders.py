import json
from sklearn import datasets
import torch, torchvision
from torch.utils.data import Subset
from torchvision import datasets, transforms
import sklearn
import ricci_regularization
from tqdm.notebook import tqdm

def get_dataloaders(dataset_config: dict, data_loader_config: dict, dtype: str, datasets_root = None):
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

    elif dataset_config["name"] in ["MNIST_subset","MNIST01"]:
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
    if dataset_config["name"] in ["MNIST01","MNIST","MNIST_subset"]:
        loaders["test_dataset"] = test_dataset
        loaders["test_dataset_partial"] = test_data
    else:
        loaders["test_dataset"] = test_data
    return loaders


def get_tuned_nn(config: dict, additional_path = '', verbose = True):
    # this function returns the initialized and loaded model  and the path to the model weights
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


# this is very oldstyle. to be deprecated soon
def get_dataloaders_tuned_nn(Path_experiment_json:str, additional_path = ''):
    """
    Loads dataset, splits it into training and testing sets, creates DataLoaders, sets up a VAE architecture, 
    loads pre-trained weights, and returns a dictionary containing the DataLoaders, the tuned VAE, and the configuration.

    Parameters:
    Path_experiment_json (str): Path to the JSON configuration file.
    additional_path (str, optional): Additional path to the pre-trained weights.

    Returns:
    dict: A dictionary with the training DataLoader, testing DataLoader, the tuned AE model, and the configuration.
    """

    with open(Path_experiment_json) as json_file:
        json_config = json.load(json_file)

    Path_experiments = json_config["Path_experiments"]
    experiment_name = json_config["experiment_name"]
    experiment_number = json_config["experiment_number"]
    Path_pictures = json_config["Path_pictures"]
    dataset_name    = json_config["dataset"]["name"]
    split_ratio = json_config["optimization_parameters"]["split_ratio"]
    batch_size  = json_config["optimization_parameters"]["batch_size"]
    datasets_root = '../../datasets/'
    # Dataset uploading 

   
    if dataset_name == "MNIST":
        #MNIST_SIZE = 28
        # MNIST Dataset
        D = 784
        train_dataset = datasets.MNIST(root=datasets_root, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset  = datasets.MNIST(root=datasets_root, train=False, transform=transforms.ToTensor(), download=False)
    elif dataset_name == "MNIST01":
        D = 784
        full_mnist_dataset = datasets.MNIST(root=datasets_root, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset  = datasets.MNIST(root=datasets_root, train=False, transform=transforms.ToTensor(), download=False)
        #indices01 = torch.where((full_mnist_dataset.targets == 0) | (full_mnist_dataset.targets == 1))[0]
        mask = (full_mnist_dataset.targets == -1) 
        selected_labels = json_config["dataset"]["selected_labels"]
        for label in selected_labels:
            mask = mask | (full_mnist_dataset.targets == label)
        indices01 = torch.where(mask)[0]
        train_dataset = Subset(full_mnist_dataset, indices01) # MNIST only with 0,1 indices

    elif dataset_name == "Synthetic":
        k = json_config["dataset"]["parameters"]["k"]
        n = json_config["dataset"]["parameters"]["n"]
        d = json_config["architecture"]["latent_dim"]
        D = json_config["architecture"]["input_dim"]
        shift_class = json_config["dataset"]["parameters"]["shift_class"]
        interclass_variance = json_config["dataset"]["parameters"]["interclass_variance"]
        variance_of_classes = json_config["dataset"]["parameters"]["variance_of_classes"]
        # Generate dataset
        # via classes
        torch.manual_seed(0) # reproducibility
        my_dataset = ricci_regularization.SyntheticDataset(k=k,n=n,d=d,D=D,
                                            shift_class=shift_class, interclass_variance=interclass_variance, variance_of_classes=variance_of_classes)

        train_dataset = my_dataset.create
    elif dataset_name == "Swissroll":
        swissroll_noise = json_config["dataset"]["parameters"]["swissroll_noise"]
        sr_numpoints = json_config["dataset"]["parameters"]["sr_numpoints"]

        D = 3
        train_dataset =  sklearn.datasets.make_swiss_roll(n_samples=sr_numpoints, noise=swissroll_noise)
        sr_points = torch.from_numpy(train_dataset[0]).to(torch.float32)
        #sr_points = torch.cat((sr_points,torch.zeros(sr_numpoints,D-3)),dim=1)
        sr_colors = torch.from_numpy(train_dataset[1]).to(torch.float32)
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(sr_points,sr_colors)

    m = len(train_dataset)
    train_data, test_data = torch.utils.data.random_split(train_dataset, [m-int(m*split_ratio), int(m*split_ratio)])

    test_loader  = torch.utils.data.DataLoader(test_data , batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # AE structure
    latent_dim = json_config["architecture"]["latent_dim"]
    input_dim  = json_config["architecture"]["input_dim"]
    try:
        architecture_type = json_config["architecture"]["name"]
    except KeyError:
        architecture_type = "TorusAE"
    if architecture_type== "TorusAE":
        torus_ae   = ricci_regularization.Architectures.TorusAE(x_dim=input_dim, h_dim1= 512, h_dim2=256, z_dim=latent_dim)
    elif architecture_type =="TorusConvAE":
        torus_ae   = ricci_regularization.Architectures.TorusConvAE(x_dim=input_dim, h_dim1= 512, h_dim2=256, z_dim=latent_dim,pixels=28)
    if torch.cuda.is_available():
        torus_ae.cuda()

    #PATH_ae_wights = "../" + json_config["weights_saved_at"]
    try:
        PATH_ae_wights = additional_path + json_config["weights_saved_at"]
    except KeyError:
        PATH_ae_wights = additional_path + json_config["weights_file"]
    torus_ae.load_state_dict(torch.load(PATH_ae_wights))
    torus_ae.eval()
    
    dict = {
        "train_loader" : train_loader,
        "test_loader" : test_loader,
        "tuned_neural_network": torus_ae,
        "json_config" : json_config
    }
    return dict


def get_dataset_and_encoding_from_dataloader(dataloader, encoder, input_dim):
    """
    Extracts and returns the input data, latent encodings from the encoder,
    and associated labels from a given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): Batch iterator over (data, labels).
        encoder (nn.Module): Encoder network that projects inputs to latent space.
        input_dim (int): Flattened dimensionality of input (e.g., 784 for MNIST).

    Returns:
        dict: {
            "input_dataset": Raw input data tensor of shape [N, input_dim],
            "AE_latent_encoding": Latent encodings (detached),
            "labels": Corresponding labels (detached)
        }
    """
    colorlist = []
    enc_list = []
    input_dataset_list = []

    for (data, labels) in tqdm( dataloader, position=0 ):
        input_dataset_list.append(data)
        enc_list.append(encoder(data.view(-1, input_dim)))
        colorlist.append(labels) 

    input_dataset = torch.cat(input_dataset_list).reshape(-1, input_dim)
    encoded_points = torch.cat(enc_list)
    encoded_points_no_grad = encoded_points.detach()
    color_array = torch.cat(colorlist).detach()
    results = {
        "input_dataset": input_dataset,
        "AE_latent_encoding": encoded_points_no_grad,
        "labels": color_array
    }
    return results