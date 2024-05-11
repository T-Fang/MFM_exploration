import os
from src.basic.constants import PREV_PHASE
from src.utils.file_utils import get_best_params_file_path, get_run_dir, combine_all_param_dicts


def get_agg_seeds_range(agg_seeds_num, agg_seed_idx):
    """
    Get the range of seeds to aggregate

    Parameters
    ----------
    agg_seeds_num : int
        The number of seeds to aggregate
    seed_idx : int
        The index of the aggregated seed

    """
    seed_start = (agg_seed_idx - 1) * agg_seeds_num + 1
    seed_end = agg_seed_idx * agg_seeds_num
    agg_seed_label = f'_seed{seed_start}_to_seed{seed_end}'
    return range(seed_start, seed_end + 1), agg_seed_label


def get_prev_phase_best_params_path(ds_name,
                                    target,
                                    phase,
                                    trial_idx,
                                    seed_idx,
                                    use_agg_seeds: bool = False,
                                    agg_seeds_num: int = 2):
    if use_agg_seeds:
        seeds_range, seed_idx = get_agg_seeds_range(agg_seeds_num, seed_idx)
    prev_phase = PREV_PHASE[phase]
    prev_phase_save_dir = get_run_dir(ds_name, target, prev_phase, trial_idx,
                                      seed_idx)
    prev_phase_best_params_path = get_best_params_file_path(
        prev_phase, prev_phase_save_dir)
    # print(prev_phase_best_params_path)
    # print(not os.path.exists(prev_phase_best_params_path))

    # if we haven't saved best params from previous phase, get it from all seeds
    if use_agg_seeds and not os.path.exists(prev_phase_best_params_path):
        prev_phase_all_seeds_save_dir = [
            get_run_dir(ds_name, target, prev_phase, trial_idx, seed_i)
            for seed_i in seeds_range
        ]
        prev_phase_all_seeds_best_params_path = [
            get_best_params_file_path(prev_phase, prev_phase_save_dir)
            for prev_phase_save_dir in prev_phase_all_seeds_save_dir
        ]
        combine_all_param_dicts(
            paths_to_dicts=prev_phase_all_seeds_best_params_path,
            combined_dict_save_path=prev_phase_best_params_path)

    return prev_phase_best_params_path


def get_curr_phase_save_dir(ds_name,
                            target,
                            phase,
                            trial_idx,
                            seed_idx,
                            use_agg_seeds: bool = False,
                            agg_seeds_num: int = 2):
    if use_agg_seeds:
        seeds_range, seed_idx = get_agg_seeds_range(agg_seeds_num, seed_idx)
    return get_run_dir(ds_name, target, phase, trial_idx, seed_idx)
