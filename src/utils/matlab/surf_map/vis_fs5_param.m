function vis_fs5_param(parameter, save_fig_path, fig_title, threshold)
    % parameter: [20484, 1]

    % load the parameter if it is a string path
    if ischar(parameter)
        parameter = csvread(parameter);
    end

    cortex_label = csvread('/home/ftian/storage/projects/MFM_exploration/src/utils/matlab/labels/cortex_fs5_label.csv');
    cortex_label = cast(cortex_label, 'int32');
    cortex_mask = cortex_label == 2;
    cortex_param = parameter(cortex_mask);

    % get the threshold if not provided
    if ~exist('threshold', 'var')
        threshold = get_colormap_threshold(cortex_param);
    end

    parameter(parameter < threshold(1)) = threshold(1);
    parameter(parameter > threshold(2)) = threshold(2);
    
    % lh_param = parameter(1:10242);
    % rh_param = parameter(10243:20484);
    
    %% Draw w, self connection
    %create new color table
    %generate a color table
    colorname = 'cool';
    HW = Num2Color(parameter,colorname);
    HW = [169 169 169; HW; 1 1 1];
    
    %% load 7-network which contains lh_labels and rh_labels, which are yeo7's labels in fsaverage5
    load('/home/shaoshi.z/storage/MFM/figure_utilities/data/1000subjects_clusters007_ref.mat');
    
    CBIG_DrawSurfaceMapsWithBoundary_mod(1:10242,10243:20484, lh_labels, rh_labels, ...
        'fsaverage5', 'inflated', 0, 20485, HW, parameter, fig_title)

    
    saveas(gcf, save_fig_path);
    close
end
