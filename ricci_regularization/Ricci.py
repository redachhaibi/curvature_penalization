import torch
import functools
from typing import Callable, Dict, Tuple, Union, Optional

class RicciCurvature:
    """
    A class for computing Riemannian curvature quantities using automatic differentiation.
    
    This class provides both forward-mode (jacfwd) and backward-mode (jacrev) differentiation
    options for computing Riemannian metrics, Christoffel symbols, and curvature tensors.
    """
    
    def __init__(self, 
                 latent_space_dim: int = 2,
                 autodiff_mode: str = "forward",
                 eps: float = 0.0,
                 device: Optional[torch.device] = None):
        """
        Initialize the RicciCurvature calculator.
        
        Parameters:
        - latent_space_dim (int): Dimensionality of the latent space (default=2)
        - autodiff_mode (str): Either "forward" or "backward" for jacfwd or jacrev (default="forward")
        - eps (float): Regularization parameter for metric inversion (default=0.0)
        - device (torch.device): Device to perform computations on (default=None, uses input tensor device)
        """
        self.latent_space_dim = latent_space_dim
        self.autodiff_mode = autodiff_mode.lower()
        self.eps = eps
        self.device = device
        
        if self.autodiff_mode not in ["forward", "backward"]:
            raise ValueError("autodiff_mode must be either 'forward' or 'backward'")
            
        # Set the jacobian function based on mode
        self.jac_func = torch.func.jacfwd if self.autodiff_mode == "forward" else torch.func.jacrev
        
        # Create vectorized versions
        self._metric_vmap = torch.func.vmap(self._compute_metric_single)
        self._christoffel_vmap = torch.func.vmap(self._compute_christoffel_single)
    
    def _compute_metric_single(self, point: torch.Tensor, function: Callable) -> torch.Tensor:
        """
        Compute the Riemannian metric (pullback metric) for a single point.
        
        Parameters:
        - point (torch.Tensor): Input point of shape (latent_space_dim,)
        - function (Callable): Function whose Jacobian defines the metric
        
        Returns:
        - torch.Tensor: The Riemannian metric tensor
        """
        point = point.reshape(-1, self.latent_space_dim)
        jac = self.jac_func(function)(point)
        jac = jac.reshape(-1, self.latent_space_dim)
        metric = torch.matmul(jac.T, jac)
        return metric
    
    def compute_metric(self, point: torch.Tensor, function: Callable) -> torch.Tensor:
        """
        Compute the Riemannian metric for single or batch of points.
        
        Parameters:
        - point (torch.Tensor): Input point(s) of shape (latent_space_dim,) or (batch_size, latent_space_dim)
        - function (Callable): Function whose Jacobian defines the metric
        
        Returns:
        - torch.Tensor: The Riemannian metric tensor(s)
        """
        if point.dim() == 1:
            return self._compute_metric_single(point, function)
        else:
            return self._metric_vmap(point, function)
    
    def _aux_func_metric(self, x: torch.Tensor, function: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Auxiliary function to return both metric and its value for use with has_aux=True.
        
        Parameters:
        - x (torch.Tensor): Input tensor
        - function (Callable): Function to compute the metric
        
        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: (metric, metric) for has_aux compatibility
        """
        g = self._compute_metric_single(x, function)
        return g, g
    
    def _compute_christoffel_single(self, point: torch.Tensor, function: Callable) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Christoffel symbols, metric tensor, and inverse metric for a single point.
        
        Parameters:
        - point (torch.Tensor): Input point where geometric quantities are evaluated
        - function (Callable): Function inducing the Riemannian metric
        
        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (Christoffel symbols, metric, inverse metric)
        """
        # Compute metric and its derivatives
        dg, g = self.jac_func(
            functools.partial(self._aux_func_metric, function=function),
            has_aux=True
        )(point)
        
        # Compute inverse of metric with regularization
        d = g.shape[0]
        device = g.device if self.device is None else self.device
        g_inv = torch.inverse(g + self.eps * torch.eye(d, device=device))
        
        # Compute Christoffel symbols
        Ch = 0.5 * (
            torch.einsum('im,mkl->ikl', g_inv, dg) +
            torch.einsum('im,mlk->ikl', g_inv, dg) -
            torch.einsum('im,klm->ikl', g_inv, dg)
        )
        
        return Ch, g, g_inv
    
    def compute_christoffel(self, point: torch.Tensor, function: Callable) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Christoffel symbols for single or batch of points.
        
        Parameters:
        - point (torch.Tensor): Input point(s)
        - function (Callable): Function inducing the Riemannian metric
        
        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (Christoffel symbols, metric, inverse metric)
        """
        if point.dim() == 1:
            return self._compute_christoffel_single(point, function)
        else:
            return self._christoffel_vmap(point, function)
    
    def _aux_func_christoffel(self, x: torch.Tensor, function: Callable) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Auxiliary function to return Christoffel symbols and additional quantities for higher-order derivatives.
        
        Parameters:
        - x (torch.Tensor): Input tensor
        - function (Callable): Function to compute Christoffel symbols
        
        Returns:
        - Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: 
          (Christoffel symbols, (Christoffel symbols, metric, inverse metric))
        """
        Ch, g, g_inv = self._compute_christoffel_single(x, function)
        return Ch, (Ch, g, g_inv)
    
    def compute_curvature_loss(self, 
                             points: torch.Tensor, 
                             function: Callable, 
                             reduction: str = "mean") -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute the curvature loss based on Riemann curvature tensor, Ricci tensor, and scalar curvature.
        
        Parameters:
        - points (torch.Tensor): Input points where curvature loss is computed
        - function (Callable): Function whose curvature is being evaluated
        - reduction (str): How to reduce the computed loss. Options:
            - "mean": Returns mean curvature loss (memory efficient for training)
            - "curvature_metric": Returns scalar curvature R and metric g
            - "dict": Returns dictionary with all computed tensors
            
        Returns:
        - Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
          Depending on reduction mode
        """
        # Compute Christoffel symbols and derivatives using vmap
        christoffel_vmap = torch.func.vmap(
            self.jac_func(functools.partial(self._aux_func_christoffel, function=function), has_aux=True)
        )
        dCh, (Ch, g, g_inv) = christoffel_vmap(points)
        
        if reduction in ["mean", "curvature_metric"]:
            # Memory-efficient computation for training
            Riemann = (torch.einsum("biljk->bijkl", dCh) - 
                      torch.einsum("bikjl->bijkl", dCh))
            Riemann += (torch.einsum("bikp,bplj->bijkl", Ch, Ch) - 
                       torch.einsum("bilp,bpkj->bijkl", Ch, Ch))
            
            # Clean up intermediate tensors for memory efficiency
            del Ch, dCh
            
            Ricci = torch.einsum("bcack->bak", Riemann)
            del Riemann
            
            R = torch.einsum('bak,bak->b', g_inv, Ricci)
            del g_inv, Ricci
            
            if reduction == "mean":
                return ((R**2) * torch.sqrt(torch.det(g))).mean()
            elif reduction == "curvature_metric":
                return R, g
                
        elif reduction == "dict":
            # Full computation for analysis/reports
            Riemann = (torch.einsum("biljk->bijkl", dCh) - 
                      torch.einsum("bikjl->bijkl", dCh))
            Riemann += (torch.einsum("bikp,bplj->bijkl", Ch, Ch) - 
                       torch.einsum("bilp,bpkj->bijkl", Ch, Ch))
            
            Ricci = torch.einsum("bcack->bak", Riemann)
            R = torch.einsum('bak,bak->b', g_inv, Ricci)
            
            return {
                "R": R,
                "g": g,
                "g_inv": g_inv,
                "Ch": Ch,
                "dCh": dCh,
                "Ricci": Ricci,
                "Riemann": Riemann
            }
        else:
            raise ValueError(f"Unknown reduction mode: {reduction}. Use 'mean', 'curvature_metric', or 'dict'")
    
    def compute_jacobian_norm(self, input_tensor: torch.Tensor, function: Callable) -> torch.Tensor:
        """
        Compute the norm of the Jacobian matrix of a function.
        
        Parameters:
        - input_tensor (torch.Tensor): Input tensor to the function
        - function (Callable): Function whose Jacobian norm is computed
        
        Returns:
        - torch.Tensor: Scalar representing the norm of the Jacobian matrix
        """
        input_tensor = input_tensor.reshape(-1, self.latent_space_dim)
        return self.jac_func(function)(input_tensor).norm()
    
    def compute_jacobian_norm_batch(self, input_tensor: torch.Tensor, function: Callable) -> torch.Tensor:
        """
        Compute Jacobian norms for a batch of points.
        
        Parameters:
        - input_tensor (torch.Tensor): Batch of input tensors
        - function (Callable): Function whose Jacobian norms are computed
        
        Returns:
        - torch.Tensor: Batch of Jacobian norms
        """
        return torch.func.vmap(self.compute_jacobian_norm)(input_tensor, function)
    
    def set_autodiff_mode(self, mode: str):
        """
        Change the autodiff mode.
        
        Parameters:
        - mode (str): Either "forward" or "backward"
        """
        if mode.lower() not in ["forward", "backward"]:
            raise ValueError("Mode must be either 'forward' or 'backward'")
        
        self.autodiff_mode = mode.lower()
        self.jac_func = torch.func.jacfwd if self.autodiff_mode == "forward" else torch.func.jacrev
        
        # Recreate vectorized functions with new mode
        self._metric_vmap = torch.func.vmap(self._compute_metric_single)
        self._christoffel_vmap = torch.func.vmap(self._compute_christoffel_single)
    
    def set_regularization(self, eps: float):
        """
        Set the regularization parameter for metric inversion.
        
        Parameters:
        - eps (float): Regularization parameter
        """
        self.eps = eps


# Legacy function wrappers for backward compatibility
def metric_jacfwd(point, function, latent_space_dim=2):
    """Legacy wrapper for forward-mode metric computation."""
    calculator = RicciCurvature(latent_space_dim=latent_space_dim, autodiff_mode="forward")
    return calculator.compute_metric(point, function)

def metric_jacrev(point, function, latent_space_dim=2):
    """Legacy wrapper for backward-mode metric computation."""
    calculator = RicciCurvature(latent_space_dim=latent_space_dim, autodiff_mode="backward")
    return calculator.compute_metric(point, function)

def curvature_loss_jacfwd(points, function, eps=0.0, reduction="mean"):
    """
    Legacy wrapper for forward-mode curvature loss computation.
    Computes the curvature loss based on the Riemann curvature tensor, Ricci tensor, and scalar curvature.
    Computation via forward propagation through jacfwd.
    When reduction is "mean" consumes less memory (auxiliary tensors are deleted - better for training)

    Parameters:
    - points: The input points where the curvature loss is computed (typically the data or latent points).
    - function: The function whose curvature is being evaluated.
    - eps: A small epsilon value used for regularization of the inverse of metric computation. (default is 0.0).
    - reduction (default = "mean"): How to reduce the computed loss ("mean" or "dict"). "mean" averages the loss, while "dict" returns the individual components.

    Returns:
    - If reduction is "mean": The mean curvature loss over the batch.
    - If reduction is "Curvature_metric": Scalar curvature R and metric g on the batch.
    - If reduction is "dict": A dictionary containing all the tensors used to compute the loss on the batch.
    """
    calculator = RicciCurvature(autodiff_mode="forward", eps=eps)
    return calculator.compute_curvature_loss(points, function, reduction)

def curvature_loss_jacrev(points, function, eps=0.0, reduction="mean"):
    """Legacy wrapper for backward-mode curvature loss computation."""
    calculator = RicciCurvature(autodiff_mode="backward", eps=eps)
    return calculator.compute_curvature_loss(points, function, reduction)

# Vectorized versions for backward compatibility
metric_jacfwd_vmap = torch.func.vmap(metric_jacfwd)
metric_jacrev_vmap = torch.func.vmap(metric_jacrev)

def Jacobian_norm_jacrev(input_tensor, function, input_dim):
    """Legacy wrapper for Jacobian norm computation."""
    calculator = RicciCurvature(latent_space_dim=input_dim, autodiff_mode="backward")
    return calculator.compute_jacobian_norm(input_tensor, function)

Jacobian_norm_jacrev_vmap = torch.func.vmap(Jacobian_norm_jacrev)


# ------------------------------------------
# various custom embeddings (use instead of the 'decoder') used foe ground truth checks 
# polynomial local diffeomorphysm of R^2
def my_fun_polinomial(u):
    u = u.flatten()
    x = u[0]
    y = u[1]

    x_out = x**2 + y + 37*x
    y_out = y**3+x*y

    x_out = x_out.unsqueeze(0)
    y_out = y_out.unsqueeze(0)
    output = torch.cat((x_out, y_out),dim=-1)
    output = output.flatten()
    return output

# Functions with sphere and Lobachevsky plane pullback metrics
# Sphere embedding
# Input: u is a 2d-vector with longitude and lattitude
# Outut: output contains the 3d coordinates of sphere and padded with zeros (781 dimension)
#        -> 784 dim in total
# u = (\theta, \phi)
# ds^2 = (d\theta)^2 + sin^2(\theta)*(d\phi)^2
def my_fun_sphere(u,D=3):
    #u = u.flatten()
    ushape = u.shape
    u = u.reshape(-1,2)
    x = torch.cos(u[:,0])*torch.cos(u[:,1])
    y = torch.cos(u[:,0])*torch.sin(u[:,1])
    z = torch.sin(u[:,0])

    output = torch.stack((x, y, z),dim=-1)
    output = output.reshape((*ushape[:-1],3))
    """
    x = torch.sin(u[:,0])*torch.cos(u[:,1])
    y = torch.sin(u[:,0])*torch.sin(u[:,1])
    z = torch.cos(u[:,0])

    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    output = torch.cat((x, y, z),dim=-1)
    """
    if D>3:
        output = torch.cat((output.unsqueeze(0),torch.zeros(D-3).unsqueeze(0)),dim=1)
    #output = output.flatten()
    return output

# Hyperbolic plane embedding
# Partial embedding (for y>c) of Lobachevsky plane to R^3 
# (formally here it is R^784)
# ds^2 = 1/y^2(dx^2 + dy^2)
# http://www.antoinebourget.org/maths/2018/08/08/embedding-hyperbolic-plane.html
def my_fun_lobachevsky(u, c=0.01):
    u = u.flatten()
    x = u[0]
    y = u[1]
    t = torch.acosh(y/c)
    x0 = t - torch.tanh(t)
    x1 = (1/torch.sinh(t))*torch.cos(x/c)
    x2 = (1/torch.sinh(t))*torch.sin(x/c)
    output = torch.cat((x0.unsqueeze(0),x1.unsqueeze(0),x2.unsqueeze(0)),dim=-1)
    output = torch.cat((output.unsqueeze(0),torch.zeros(781).unsqueeze(0)),dim=1)
    output = output.flatten()
    return output