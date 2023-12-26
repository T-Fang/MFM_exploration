import os
import subprocess
from src.basic.constants import LOG_PATH, MATLAB_SCRIPT_PATH


def get_target_path(ds_name, target):
    target_path = os.path.join(LOG_PATH, ds_name, target)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    return target_path


def get_run_path(ds_name, target, mode, trial_idx, seed_idx):
    run_path = os.path.join(get_target_path(ds_name, target), mode,
                            f'trial{trial_idx}', f'seed{seed_idx}')
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    return run_path


def get_fig_dir(ds_name, target, fig_type, trial_idx, seed_idx):
    fig_dir = os.path.join(get_target_path(ds_name, target), 'figures',
                           fig_type, f'trial{trial_idx}', f'seed{seed_idx}')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    return fig_dir


def get_fig_file_path(ds_name, target, fig_type, trial_idx, seed_idx,
                      fig_name):
    fig_dir = get_fig_dir(ds_name, target, fig_type, trial_idx, seed_idx)
    return os.path.join(fig_dir, fig_name)


def visualize_param(mat_file_path, param_name, fig_file_path):
    """
    First, cd into MATLAB_SCRIPT_PATH.
    Then, load the parameter stored in the `mat_file_path` file.
    Finally, run the visualize_parameter_desikan_fslr function to visualize the parameter.
    """

    print(f"Visualizing {param_name} from {mat_file_path}...")

    command = [
        (f"cd {MATLAB_SCRIPT_PATH}; "
         f"matlab -nodisplay -nosplash -nodesktop -r "
         f"\"load('{mat_file_path}', '{param_name}'); "
         f"visualize_parameter_desikan_fslr({param_name}, '{fig_file_path}'); "
         f"exit;\"")
    ]

    result = subprocess.run(command,
                            shell=True,
                            capture_output=True,
                            text=True)
    print(result.stdout)
    print(result.stderr)

    print(f'Visualization saved to {fig_file_path}')
