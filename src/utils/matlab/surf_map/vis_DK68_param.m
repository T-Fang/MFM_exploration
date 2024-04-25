function vis_DK68_param(parameter, save_fig_path, fig_title, threshold)
    % parameter: [68, 1]
    roi_list = ones(72, 1);
    roi_list([1 5 37 41]) = 0;
    % if threshold does not exist
    if ~exist('threshold', 'var')
        threshold = get_colormap_threshold(parameter);
    end

    WIMap_mod_roi(parameter, roi_list, fig_title, threshold);

    saveas(gcf, save_fig_path);
    close

end
    


function WIMap_mod_roi(Para_E, roi_list, fig_title, threshold)

    %--------------------------------------------------------------------------
    % CBIG_mfm_rfMRI_DrawOverlap_WI_Desikan68_Yeo7_fsaverage5
    %
    % show estimated W and I in Desikan68 parcellation of fsaverage5 with Yeo's 7-network boundary
    %
    % Written by Peng Wang under MIT license: 
    % https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    %--------------------------------------------------------------------------
    
    
    %% read data (model parameter)
    data_dir = '/home/shaoshi.z/storage/MFM/figure_utilities/data'; 
    Yeo7_label_file_name = '1000subjects_clusters007_ref.mat';
    
    Wvector = Para_E;
    % check if threshold is provided as an input
    if exist('threshold', 'var')
        Wvector(Wvector < threshold(1)) = threshold(1);
        Wvector(Wvector > threshold(2)) = threshold(2);
    end
    
    populated_vector = nan(72, 1);
    counter = 1;
    for i = 1:72
       if roi_list(i) == 1 
          populated_vector(i) = Wvector(counter);
          counter = counter + 1;
       end
    end
    populated_vector([1, 5, 37, 41]) = [];
    
    colorname = 'cool';
    
    %% load desikan68
    
    lh_mesh_fsavg = CBIG_ReadNCAvgMesh('lh', 'fsaverage5', 'inflated', 'aparc.annot');
    rh_mesh_fsavg = CBIG_ReadNCAvgMesh('rh', 'fsaverage5', 'inflated', 'aparc.annot');
    
    lh_label_desikan = lh_mesh_fsavg.MARS_label;
    rh_label_desikan = rh_mesh_fsavg.MARS_label;
    
    
    %% load 7-network 
    
    load ([data_dir '/' Yeo7_label_file_name]); % this loads lh_labels and rh_labels, which are yeo7's labels in fsaverage5
    
    lh_label_desikan_data = lh_label_desikan;
    rh_label_desikan_data = rh_label_desikan + 36;
    
    for i = 1:72
       if i < 37
           if roi_list(i) == 0
               lh_label_desikan_data(lh_label_desikan_data == i) = 0;
           end
       else
           if roi_list(i) == 0
               rh_label_desikan_data(rh_label_desikan_data == i) = 0;
           end
    
       end
    end
    
    
    %% Draw w, self connection
    %create new color table
    %generate a color table
    HW = Num2Color(Wvector,colorname);
    index = 1:1:72;
    index(~roi_list) = [];
    HW_1 = ones(72,3);
    HW_1(index,:) = HW; % size: (72, 3)
    HW_1 = [169 169 169; HW_1; 1 1 1]; % size: (74, 3)
    
    %draw
    CBIG_DrawSurfaceMapsWithBoundary_mod(lh_label_desikan_data,rh_label_desikan_data, lh_labels, rh_labels, ...
        'fsaverage5', 'inflated', 0, 73, HW_1, populated_vector, fig_title)
end
    
