function HCP_get_4_runs(sub_idx)
% sub_idx starts from 1 to 1029. index from '/home/ftian/storage/projects/MFM_exploration/data/HCPYA/1029_participants/subject_1029.mat'
% Operate on ICA-FIX data which have corresponding GSR data (exclude high
% motion runs by Ruby)
    time_points = 1200;  % the length of BOLD time series

    if ~isnumeric(sub_idx)
        sub_idx = str2double(sub_idx);
    end
    sub_idx = cast(sub_idx, 'int32');

    fprintf('sub_idx: %d\n', sub_idx);

    subject_ids = load('/home/tzeng/storage/Matlab/DELSSOME/HCP/general_matfiles/subject_1029.mat');
    subject_ids = subject_ids.subject_1029;
    subject_id = num2str(subject_ids(sub_idx));
    fprintf('subject id: %s\n', subject_id);

    save_dir = '/home/ftian/storage/projects/MFM_exploration/data/HCPYA/1029_participants/TC';
    save_path = fullfile(save_dir, sprintf('%s_bold_4_runs.mat', subject_id));
    if exist(save_path, 'file') == 2
        fprintf('4 runs of subject %s already saved!\n', subject_id);
        return
    end

    lh_label = load('/home/tzeng/storage/Matlab/HCPS1200/matfiles/lh_desikan_label.mat');
    rh_label = load('/home/tzeng/storage/Matlab/HCPS1200/matfiles/rh_desikan_label.mat');
    lh_label = lh_label.lh_desikan_fslr_32k_label;
    rh_label = rh_label.rh_desikan_fslr_32k_label;

    run_list = ["REST1_LR" "REST1_RL" "REST2_LR" "REST2_RL"];

    run_num = length(run_list);
    bold_TC = NaN(run_num, 68, 1200);
    valid_run = 0;

    for session_i = 1:run_num
        run = run_list(session_i);
        fprintf("%s: ", run);
        
        fmri_gsr = sprintf(['/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/individuals/%s/MNINonLinear/Results/rfMRI_%s' ...
            '/postprocessing/MSM_reg_wbsgrayordinatecortex/rfMRI_%s_Atlas_MSMAll_hp2000_clean_regress.dtseries.nii'], subject_id, run, run);
        if exist(fmri_gsr, 'file') ~= 2
            continue
        end
        fmri_ica = sprintf(['/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/individuals/%s/MNINonLinear/Results/rfMRI_%s'...
            '/rfMRI_%s_Atlas_MSMAll_hp2000_clean.dtseries.nii'], subject_id, run, run);
        bold_raw = ft_read_cifti(fmri_ica);
        bold_raw = bold_raw.dtseries;
        if size(bold_raw, 2) ~= time_points
            fprintf("The dtseries' length is not %d.\n", time_points);
            continue
        end
        
        fprintf("Valid run.\n");
        valid_run = valid_run + 1;
        bold_TC(valid_run, :, :) = convert_TC_fslr2DK68(bold_raw(1:64984, :), lh_label, rh_label);
        % [roi_num, t_len]

    end

    % save the bold_TC
     
    if valid_run == run_num
        save_path = fullfile(save_dir, sprintf('%s_bold_4_runs.mat', subject_id));
        save(save_path, 'bold_TC');
    else
        fprintf('subject %s only has %d/4 runs\n', subject_id, valid_run);
        bold_TC = bold_TC(1:valid_run, :, :);
        save_path = fullfile(save_dir, sprintf('%s_bold_valid_runs.mat', subject_id));
        save(save_path, 'bold_TC');
    end