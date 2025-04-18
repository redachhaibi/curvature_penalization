import torch
import functools

# notations have to be clearer use full names

# Jacobian norm for contractive loss computation
def Jacobian_norm_jacrev(input_tensor, function, input_dim):
    """
    Computes the norm of the Jacobian matrix of a function using reverse-mode autodiff (jacrev).

    Parameters:
    - input_tensor (torch.Tensor): The input tensor to the function.
    - function (Callable): The function whose Jacobian is to be computed.
    - input_dim (int): Dimensionality of the input space.

    Returns:
    - torch.Tensor: Scalar representing the norm of the Jacobian matrix.
    """
    input_tensor = input_tensor.reshape(-1,input_dim)
    return torch.func.jacrev(function)(input_tensor).norm()

Jacobian_norm_jacrev_vmap = torch.func.vmap(Jacobian_norm_jacrev)

# Forward mode propagation via jacfwd

def metric_jacfwd(u, function, latent_space_dim=2):
    """
    Computes the Riemannian metric (pullback metric) induced by a function (e.g., decoder) using forward-mode autodiff.

    Parameters:
    - u (torch.Tensor): Input point(s) of shape (latent_space_dim,) or batch of points.
    - function (Callable): Function (typically decoder) whose Jacobian defines the metric.
    - latent_space_dim (int): Dimensionality of the latent space (default=2).

    Returns:
    - torch.Tensor: The Riemannian metric (symmetric positive semi-definite matrix) at the input point.
    """
    u = u.reshape(-1,latent_space_dim)
    jac = torch.func.jacfwd(function)(u)
    jac = jac.reshape(-1,latent_space_dim)
    metric = torch.matmul(jac.T,jac)
    return metric

metric_jacfwd_vmap = torch.func.vmap(metric_jacfwd)

# this function is auxiliary in computing metric and its derivatives later
# as one needs to output both the result and its derivative simultanuousely 
def aux_func_metric(x, function):
    """
    Auxiliary function used to return both the Riemannian metric and its value for use with jacfwd.

    Parameters:
    - x (torch.Tensor): Input tensor.
    - function (Callable): Function (typically decoder) to compute the metric.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: The metric tensor and a copy (for use with `has_aux=True` in `jacfwd`).
    """
    g = metric_jacfwd( x, function=function)
    return g, g

# this also not vectorized
def Ch_g_g_inv_jacfwd (u, function, eps = 0.0):
    """
    Computes Christoffel symbols, metric tensor, and inverse metric using forward-mode autodiff.

    Parameters:
    - u (torch.Tensor): Input point where the geometric quantities are evaluated.
    - function (Callable): Function (typically decoder) inducing the Riemannian metric.
    - eps (float): Small value added to the diagonal for numerical stability (default=0.0).

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - Christoffel symbols: tensor of shape (dim, dim, dim)
        - Metric tensor: shape (dim, dim)
        - Inverse metric tensor: shape (dim, dim)
    """
    # compute metric and its derivatives at a batch of points
    dg, g = torch.func.jacfwd( functools.partial(aux_func_metric, function=function),
                         has_aux=True)( u )
    # compute inverse of metric with some regularization param eps    
    d = g.shape[0]
    device = g.device
    g_inv = torch.inverse(g + eps*torch.eye(d,device=device))
    # compute Christoffel symbols
    Ch = 0.5*(torch.einsum('im,mkl->ikl',g_inv,dg)+
              torch.einsum('im,mlk->ikl',g_inv,dg)-
              torch.einsum('im,klm->ikl',g_inv,dg)
              )
    return Ch, g, g_inv

def aux_func(x,function, eps=0.0):
    """
    Auxiliary function to return Christoffel symbols and additional metric quantities for use in higher-order derivatives.

    Parameters:
    - x (torch.Tensor): Input tensor.
    - function (Callable): Decoder or similar function.
    - eps (float): Regularization parameter for metric inversion (default=0.0).

    Returns:
    - Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        - Christoffel symbols,
        - Tuple of (Christoffel symbols, metric, inverse metric).
    """
    Ch, g, g_inv = Ch_g_g_inv_jacfwd( x, function=function, eps=eps)
    return Ch, (Ch, g, g_inv)

# computing the loss
def curvature_loss_jacfwd (points, function, eps = 0.0, reduction = "mean"):
    """
    Computes the curvature loss based on the Riemann curvature tensor, Ricci tensor, and scalar curvature.
    Computation via forward propagation through jacfwd.

    Parameters:
    - points: The input points where the curvature loss is computed (typically the data or latent points).
    - function: The function whose curvature is being evaluated.
    - eps: A small epsilon value used for regularization of the inverse of metric computation. (default is 0.0).
    - reduction (default = "mean"): How to reduce the computed loss ("mean" or "dict"). "mean" averages the loss, while "dict" returns the individual components.

    Returns:
    - If reduction is "mean": The mean curvature loss over the batch.
    - If reduction is "dict": A dictionary containing the individual components used to compute the loss.
    """
    # compute Christoffel symbols and derivatives and inverse of metric
    dCh, (Ch, g, g_inv) = torch.func.vmap( torch.func.jacfwd(functools.partial( aux_func, function=function, eps=eps),
                            has_aux=True) )( points )
    
    Riemann = torch.einsum("biljk->bijkl",dCh) - torch.einsum("bikjl->bijkl",dCh)
    Riemann += torch.einsum("bikp,bplj->bijkl", Ch, Ch) - torch.einsum("bilp,bpkj->bijkl", Ch, Ch)
    
    Ricci = torch.einsum("bcack->bak",Riemann)
    R = torch.einsum('bak,bak->b',g_inv,Ricci)
    if reduction == "mean":
        return ( ( R**2 ) * torch.sqrt( torch.det(g) ) ).mean()
    elif reduction == "dict":
        dict = {
            "R": R,
            "g": g,
            "g_inv": g_inv,
            "Ch": Ch,
            "dCh": dCh,
            "Ricci": Ricci
        }
        return dict

# computation of Scalar curvature and metric
# redundunt with curvature_loss_jacfwd has to be substituted in ipynb files and deprecated.
def Sc_g_jacfwd (u, function, eps = 0.0):
    """
    Returns scalar curvature and metric, calls the domputation of the Riemann curvature tensor and Ricci tensor. 
    Using forward-mode Jacobian derivatives.
    
    Parameters:
    - u: The input points (typically data or latent points).
    - function(decoder): The function (usually the decoder) whose gradient(Jacobian) defines the metric and curvature is being evaluated.
    - eps: A small epsilon value used for regularization of the inverse of metric computation. (default is 0.0).

    Returns:
    - Sc: The scalar curvature computed from the Jacobian of the function at point u.
    - g: The metric computed from the Jacobian of the function at point u.
    """
    # compute Christoffel symbols and derivatives and inverse of metric
    dCh, (Ch, g, g_inv) = torch.func.jacfwd(functools.partial( aux_func, function=function, eps=eps),
                            has_aux=True)( u )
    
    Riemann = torch.einsum("iljk->ijkl",dCh) - torch.einsum("ikjl->ijkl",dCh)
    Riemann += torch.einsum("ikp,plj->ijkl", Ch, Ch) - torch.einsum("ilp,pkj->ijkl", Ch, Ch)
    
    Ricci = torch.einsum("cacb->ab",Riemann)
    Sc = torch.einsum('ab,ab',g_inv,Ricci)
    return Sc, g
# vectorization
Sc_g_jacfwd_vmap = torch.func.vmap(Sc_g_jacfwd)

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



# Bacward mode propagation via torch.func.jacrev 
# ------------------------------------------------
# code below this line has to be rewritten as it contains recursive hell
# however it is not used in curvature computation.
"""
def metric_jacrev(u, function):
    jac = torch.func.jacrev(function)(u).squeeze()
    # squeezing is needed to get rid of 1-dimentions
    metric = torch.matmul(jac.T,jac)
    return metric

"""
def metric_jacrev(u, function, latent_space_dim=2):
    u = u.reshape(-1,latent_space_dim)
    jac = torch.func.jacrev(function)(u)
    jac = jac.reshape(-1,latent_space_dim)
    metric = torch.matmul(jac.T,jac)
    return metric


metric_jacrev_vmap = torch.func.vmap(metric_jacrev)

def metric_der_jacrev (u, function):
    metric = functools.partial(metric_jacrev, function=function)
    dg = torch.func.jacrev(metric)(u).squeeze()
    # squeezing is needed to get rid of 1-dimentions 
    # occuring when using torch.func.jacrev
    return dg
metric_der_jacrev_vmap = torch.func.vmap(metric_der_jacrev)

def Ch_jacrev (u, function):
    g = metric_jacrev(u,function)
    g_inv = torch.inverse(g)
    dg = metric_der_jacrev(u,function)
    Ch = 0.5*(torch.einsum('im,mkl->ikl',g_inv,dg)+
              torch.einsum('im,mlk->ikl',g_inv,dg)-
              torch.einsum('im,klm->ikl',g_inv,dg)
              )
    return Ch
Ch_jacrev_vmap = torch.func.vmap(Ch_jacrev)

def Ch_der_jacrev (u, function):
    Ch = functools.partial(Ch_jacrev, function=function)
    dCh = torch.func.jacrev(Ch)(u).squeeze()
    return dCh

Ch_der_jacrev_vmap = torch.func.vmap(Ch_der_jacrev)

# Riemann curvature tensor (3,1)
def Riem_jacrev(u, function):
    Ch = Ch_jacrev(u, function)
    Ch_der = Ch_der_jacrev(u, function)

    Riem = torch.einsum("iljk->ijkl",Ch_der) - torch.einsum("ikjl->ijkl",Ch_der)
    Riem += torch.einsum("ikp,plj->ijkl", Ch, Ch) - torch.einsum("ilp,pkj->ijkl", Ch, Ch)
    return Riem

def Ric_jacrev(u, function):
    Riemann = Riem_jacrev(u, function)
    Ric = torch.einsum("cacb->ab",Riemann)
    return Ric

Ric_jacrev_vmap = torch.func.vmap(Ric_jacrev)

def Sc_jacrev (u, function):
    metric = metric_jacrev(u, function=function)
    Ricci = Ric_jacrev(u, function=function)
    metric_inv = torch.inverse(metric)
    Sc = torch.einsum('ab,ab',metric_inv,Ricci)
    return Sc
Sc_jacrev_vmap = torch.func.vmap(Sc_jacrev)


    
