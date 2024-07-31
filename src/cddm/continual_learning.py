"""This module contains functions that are related to
the continual learning procedure.


Functions
---------
continual_learning
    A function for continual learning of the given sequence of tasks.
"""


import torch

from .train import train
from .pruning import gru_pruning
from .utils import loss_func, error_func, process_data


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def continual_learning(net, file_names, alpha, optimizer_name,
                       scheduler, n_epochs, lr, weight_decay,
                       device, nums_train=[800, 100, 100, 100],
                       num_val=100, num_test=100,
                       SCALE=False, seed=0, path_to_save="model.pth",
                       result_folder='./result', problem='plasticity-plates'):
    """ A function for continual learning of the given sequence of tasks.
    Parameters
    ----------

    net : PyTorch model
    The learnable GRU model.

    file_names : list
    The list of input files.

    alpha : float
    Pruning parameter between 0 and 1.
    The higher value the less aggressive the pruning.

    optimizer_name : str
    Optimizer name.

    scheduler : torch.optim.lr_scheduler
    Scheduler for learning rate planning.

    n_epochs : int
    The number of training epochs.

    lr : float
    Learning rate.

    weight_decay : float
    Weight decay.

    device: torch.device ('cpu' or 'cuda')
    The device on which PyTorch model and all torch.
    Tensor are or will be allocated.

    nums_train : list
    List of the number of training paths for each task.

    num_val, num_test : int, int
    The number of validation/test paths.

    SCALE : bool
    Scale data or not.

    seed : int
    Random seed.

    path_to_save : str
    Paths for saving the model.

    result_folder : str
    Path for result folder.

    problem : str
    Type of the problem

    Returns
    -------
    net : PyTorch model
    The network where FC layer is pruned for the task number task_id.
    """

    num_tasks = len(file_names)
    print(file_names)

    for task_id in range(num_tasks):
        if nums_train[task_id] > 1000:
            print('ERROR: Number of training paths too large')
            return
        if nums_train[task_id] < 1:
            print('ERROR: Number of training paths must be positive')
            return

    for task_id in range(num_tasks):
        num_train = nums_train[task_id]

        print('TRAIN PATHS: ', num_train)

        x_train, y_train, x_val, y_val, x_test, y_test, _, _, y_mean, y_std = process_data(file_names[task_id],
                                                                                           num_train=num_train,
                                                                                           num_val=num_val,
                                                                                           num_test=num_test,
                                                                                           problem=problem)

        print('--------------TASK {}---------------------'.format(task_id+1))

        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=lr, weight_decay=weight_decay)

        net.set_task(task_id)
        net.set_trainable_masks(task_id)

        net = train(net=net,
                    x_train=x_train, y_train=y_train,
                    x_val=x_val, y_val=y_val,
                    device=device,
                    lr=lr, n_epochs=n_epochs,
                    optimizer=optimizer, scheduler=scheduler,
                    task_id=task_id,
                    path_to_save=path_to_save)

        net.load_state_dict(torch.load(f"{result_folder}/{path_to_save}"))

        net.eval()
        y_pred = net(x_test)
        print("test loss: ", loss_func(y_test, y_pred).item())
        print("error: %.3f" %
              (100*error_func(y_test*y_std + y_mean, y_pred*y_std +
                              y_mean).item()) + "%")
        print("------------------")

        if num_tasks > 1:
            net = gru_pruning(net, alpha, x_train, task_id, device)

            net.set_trainable_masks(task_id)

            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr, weight_decay=1e-6)
            net = train(net=net,
                        x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val,
                        device=device,
                        lr=lr, n_epochs=n_epochs//5,
                        optimizer=optimizer, scheduler=scheduler,
                        task_id=task_id,
                        path_to_save=path_to_save)

            net.load_state_dict(torch.load(f"{result_folder}/{path_to_save}"))

            net.eval()
            y_pred = net(x_test)
            print("test loss: ", loss_func(y_test, y_pred).item())
            print("error: %.3f" %
                  (100*error_func(y_test*y_std + y_mean, y_pred*y_std +
                                  y_mean).item()) + "%")
            print("------------------")

            net.set_masks_intersection()
            net.set_masks_union()

            net.save_masks(f"{result_folder}/masks_" + path_to_save)

            if task_id + 1 < num_tasks:
                net.add_mask(task_id=task_id+1)

    return net
