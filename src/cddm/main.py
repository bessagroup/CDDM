# Standard
import argparse
import os
# Third-party
import torch
import pandas as pd
# Local
from .utils import gru_total_params_mask, set_seed
from .continual_learning import continual_learning
from .models import GRU

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def main():
    """ The method is the main routine:
    it parses the input arguments,
    defines the hyperparameters and
    calls the other relevant methods.

        Parameters
        ----------
        """
    ################
    # Arguments
    ################
    parser = argparse.ArgumentParser(\
        description='Characterizing/Rethinking PINNs')

    parser.add_argument('--problem', type=str, \
                        default='plasticity-plates', help='Problem to solve.')
    parser.add_argument('--model_name', type=str, \
                        default='gru', help='Model to use.')
    parser.add_argument('--tasks', type=str, \
                        default='A,B,C,D', help='Tasks to learn.')
    parser.add_argument('--nums_train', type=str, \
                        default='800, 100, 100, 100', \
                        help='Number of training paths.')
    parser.add_argument('--data_folder', type=str, \
                        default='./data/plates', \
                        help='Path to folder with data.')

    parser.add_argument('--input_size', type=int, \
                        default=3, help='Number of input neurons.')
    parser.add_argument('--output_size', type=int, \
                        default=3, help='Number of output neurons.')
    parser.add_argument('--num_grucells', type=int, \
                        default=2, help='Number of GRU cells.')
    parser.add_argument('--hidden_size', type=int, \
                        default=128, \
                        help='Number of features in the hidden state.')

    parser.add_argument('--seq_len', type=int, \
                        default=101, help='Data sequence length.')

    parser.add_argument('--optimizer_name', type=str, \
                        default='Adam', help='Optimizer of choice.')
    parser.add_argument('--lr', type=float, \
                        default=1e-2, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, \
                        default=1e-6, help='Weight decay.')
    parser.add_argument('--n_epochs', type=int, \
                        default=1000, help='Number of training epochs.')
    parser.add_argument('--alpha', type=float, \
                        default=0.95, help='Pruning parameter')

    parser.add_argument('--save_model', action='store_true', \
                        help='Save the model.')
    parser.add_argument('--save_result', action='store_true', \
                        help='Save the results.')
    parser.add_argument('--result_folder', type=str, \
                        default='./result', help='Path to save the results.')

    parser.add_argument('--seed', type=int, \
                        default=0, help='Random initialization.')

    args = parser.parse_args()

    problem = args.problem
    model_name = args.model_name
    data_folder = args.data_folder
    result_folder = args.result_folder

    tasks = [task.strip() for task in args.tasks.split(',')]

    if problem == 'plasticity-plates':
        file_names = [f'{data_folder}/{task}.pkl' for task in tasks]
    else:
        file_names = [f'{data_folder}/{task}.pickle' for task in tasks]

    # num_tasks = len(file_names)

    SAVE_MODELS = args.save_model
    SAVE_DATA = args.save_result

    input_size = args.input_size
    output_size = args.output_size
    seq_len = args.seq_len
    hidden_size = args.hidden_size
    num_layers = args.num_grucells

    num_val = 100
    num_test = 100

    optimizer_name = args.optimizer_name
    lr = args.lr
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    alpha = args.alpha

    seed = args.seed

    all_losses = {}
    all_errors = {}

    nums_train = [int(num_train.strip()) \
                  for num_train in args.nums_train.split(',')]

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    print(f"################## SEED {seed} ##################")
    set_seed(seed)

    # for num_train in nums_train:
    net = GRU(input_size=input_size, seq_len=seq_len, hidden_size=hidden_size,
              num_layers=num_layers, \
              output_size=output_size, device=device).to(device)

    print('Total params: ', gru_total_params_mask(net))

    if SAVE_MODELS:
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        dict_names = {}
        case_name = f'{problem}_'
        for i, task in enumerate(tasks):
            case_name += f'{task}-'

        case_name = case_name[:-1]

        path_to_save = f"{model_name}_\
            {case_name}_seed{seed}_num-layers{num_layers}_\
            hidden{hidden_size}_lr0.01_alpha{alpha}.pth"
    else:
        path_to_save = f"model.pth"

    print(f">>>>>>>>>>>>{nums_train} TRAINING POINTS<<<<<<<<<<<<<<<")

    net = continual_learning(net,
                             file_names=file_names, alpha=alpha,
                             optimizer_name=optimizer_name, scheduler=None,
                             n_epochs=n_epochs, lr=lr,
                             weight_decay=weight_decay,
                             device=device,
                             nums_train=nums_train,
                             num_val=num_val,
                             num_test=num_test,
                             seed=seed,
                             path_to_save=path_to_save,
                             result_folder=result_folder,
                             problem=problem)

    losses, errors = eval(net, file_names, nums_train=nums_train,
                          num_val=num_val,
                          num_test=num_test, problem=problem)

    for i in range(len(nums_train)):
        all_losses[f"{tasks[i]}_{nums_train[i]}points"] = losses[i]
        all_errors[f"{tasks[i]}_{nums_train[i]}points"] = errors[i]

    # df_losses = pd.DataFrame([all_losses])
    df_errors = pd.DataFrame([all_errors])

    print(df_errors)

    if SAVE_DATA:
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        dict_names = {}
        case_name = f'{problem}_'
        for i, task in enumerate(tasks):
            dict_names[i] = f'{task}'
            case_name += f'{task}-'

        case_name = case_name[:-1]
        df_errors.rename(index=dict_names, inplace=True)

        if len(tasks) > 1:
            df_errors.to_csv(f"{result_folder}/{case_name}_\
                             error_seed{seed}_num-layers{num_layers}_\
                             hidden{hidden_size}_\
                             lr{lr}_alpha{alpha}.csv",
                             index=False)
        else:
            df_errors.to_csv(f"{result_folder}/{case_name}_\
                             error_seed{seed}_\
                             num-layers{num_layers}_\
                             hidden{hidden_size}_lr{lr}.csv",
                             index=False)

    if os.path.exists(f"{result_folder}/model.pth"):
        os.remove(f"{result_folder}/model.pth")

    if os.path.exists(f"{result_folder}/masks_model.pth"):
        os.remove(f"{result_folder}/masks_model.pth")

    return

if __name__ == "__main__":
    main()
