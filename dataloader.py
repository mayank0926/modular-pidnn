import torch
import numpy as np
from pyDOE import lhs


def train_loader(config):
    """Loads training dataset.

    Detailed description.

    Args:
        config: TODO

    Returns:
        dict: TODO
    """
    device = torch.device('cuda' if torch.cuda.is_available()
                          and config['cuda_enabled'] else 'cpu')

    N_u = config['num_datadriven']
    N_f = config['num_collocation']

    N_inp = config['NUM_INPUTS']
    N_out = config['NUM_OUTPUTS']
    N_init = config['NUM_INIT']

    data = np.genfromtxt(
        config['datadir'] + config['datafile'], delimiter=',', dtype=np.float32)
    N_data = data.shape[0]

    lb = np.min(data[:, :N_inp], axis=0)
    ub = np.max(data[:, :N_inp], axis=0)

    idx_init = np.array([])
    for col in config['COLS_INIT']:
        # Takes N_init values from each column specified in cols_init
        vals = np.unique(data[:, col])
        for val in vals[:min(N_init, vals.shape[0])]:
            idx_val = [data[:, col] == val]
            idx_init = np.hstack([idx_init, idx_val])

    idx_non_init = np.random.choice(np.delete(np.arange(N_data), idx_init),
                                    max(N_u - idx_init.shape[0], 0), replace=False)
    idx_train = np.hstack([idx_init, idx_non_init])
    idx_validation = np.random.choice(np.delete(np.arange(N_data), idx_train),
                                      N_u, axis=0)

    inp_u = data[idx_train, :N_inp]
    out_u = data[idx_train, -N_out:]
    inp_f = np.vstack([lb + (ub-lb)*lhs(N_inp, N_f) if N_f > 0
                      else np.array([]).reshape(0, N_inp),
                      inp_u])
    val_inp = data[idx_validation, :N_inp]
    val_out = data[idx_validation, -N_out:]

    if N_init > 1:
        out_u += config['noise'] * \
            np.std(out_u) * np.random.randn(*out_u.shape)

    return [torch.from_numpy(x).float().to(device)
            for x in [inp_u, out_u, inp_f, val_inp, val_out, lb, ub]]


def test_loader(config):
    pass
