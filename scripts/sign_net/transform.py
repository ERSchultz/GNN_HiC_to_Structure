# Transform for extracting eigenvector and eigenvalues
import torch
from torch_geometric.utils import get_laplacian, to_undirected
from torch_scatter import scatter_add
from torch_sparse import SparseTensor


# The needed pretransform to save result of EVD
class EVDTransform(object):
    def __init__(self, norm=None, k=None):
        super().__init__()
        self.norm = norm
        self.k = k
    def __call__(self, data):
        D, V = EVD_Laplacian(data, self.norm, self.k)
        data.eigen_values = D
        data.eigen_vectors = V.reshape(-1) # reshape to 1-d to save
        return data

def EVD_Laplacian(data, norm=None, k=None):
    L_raw = get_laplacian(to_undirected(data.edge_index, num_nodes=data.num_nodes),
                          normalization=norm, num_nodes=data.num_nodes)
    L = SparseTensor(row=L_raw[0][0], col=L_raw[0][1], value=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()

    D, V  = torch.linalg.eigh(L)

    if k is not None:
        D = D[:k]
        V = V[:, :k]

    return D, V


def to_dense_EVD(eigS, eigV, batch, k):
    batch_size = int(batch.max()) + 1
    num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0, dim_size=batch_size)
    max_num_nodes = int(num_nodes.max())
    if k is None:
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
        idx = torch.arange(batch.size(0), dtype=torch.long, device=eigS.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        eigS_dense = eigS.new_full([batch_size * max_num_nodes], 0)
        eigS_dense[idx] = eigS
        eigS_dense = eigS_dense.view([batch_size, max_num_nodes])

        mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=eigS.device)
        mask[idx] = 1
        mask = mask.view(batch_size, max_num_nodes)
        mask_squared = mask.unsqueeze(2) * mask.unsqueeze(1)
        eigV_dense = eigV.new_full([batch_size * max_num_nodes * max_num_nodes], 0)
        eigV_dense[mask_squared.reshape(-1)] = eigV
        eigV_dense = eigV_dense.view([batch_size, max_num_nodes, max_num_nodes])
    else:
        eigS_dense = eigS.reshape(batch_size, k)
        mask = None

        assert len(torch.unique(num_nodes)) == 1, 'only support fixed size graphs'
        eigV_dense = eigV.reshape(batch_size, max_num_nodes, k)

    # eigS_dense: B x N_max
    # eigV_dense: B x N_max x k
    return eigS_dense, eigV_dense, mask


def to_dense_list_EVD(eigS, eigV, batch, k = None):
    eigS_dense, eigV_dense, mask = to_dense_EVD(eigS, eigV, batch, k)

    nmax = eigV_dense.size(1)
    eigS_dense = eigS_dense.unsqueeze(1).repeat(1, nmax, 1)
    if mask is not None:
        eigS_dense = eigS_dense[mask]
        eigV_dense = eigV_dense[mask]
    else:
        # must be the case that all graphs have same num_nodes
        eigS_dense = eigS_dense.reshape(-1, k)
        eigV_dense = eigV_dense.reshape(-1, k)

    return eigS_dense, eigV_dense
