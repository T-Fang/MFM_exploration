function visualize_parameter_desikan_fslr(parameter, save_fig_path, H)
% parameter: [68, 1]
% H: colormap, optional.

all_desikan_label = desikan_label_subcortical_removal();
vertice_para = roi2vertices(parameter, all_desikan_label);
colormap_name = 'cool';
if(~exist('H', 'var'))
    H = Num2Color(parameter, colormap_name);
    H = [0 0 0; H];
end
min_thres = min(parameter);
max_thres = max(parameter);
h_fig = CBIG_DrawSurfaceMaps_fslr(vertice_para(1:32492), vertice_para(32493:end), 'fs_LR_32k', 'very_inflated', ...
                               min_thres - 0.01 * (max_thres - min_thres), max_thres, H);
% h_fig = CBIG_DrawSurfaceMaps_fslr(vertice_para(1:32492), vertice_para(32493:end), 'fs_LR_32k', 'very_inflated', ...
%                                -14.5e-04, -4e-04, H);
saveas(h_fig, save_fig_path);
close

end


function all_desikan_label = desikan_label_subcortical_removal()
% Subcortical will be set to 0.
    lh_desikan_label = load('/home/ftian/storage/projects/MFM_exploration/visualization/tzeng/HCPYA/matfiles/lh_desikan_label.mat');
    lh_desikan_label = lh_desikan_label.lh_desikan_fslr_32k_label;
    rh_desikan_label = load('/home/ftian/storage/projects/MFM_exploration/visualization/tzeng/HCPYA/matfiles/rh_desikan_label.mat');
    rh_desikan_label = rh_desikan_label.rh_desikan_fslr_32k_label;
    for i = 1:36
        if i == 1 || i == 5
            lh_desikan_label(lh_desikan_label==i) = 0;
            rh_desikan_label(rh_desikan_label==i) = 0;
        elseif i < 5
            lh_desikan_label(lh_desikan_label==i) = lh_desikan_label(lh_desikan_label==i) - 1;
            rh_desikan_label(rh_desikan_label==i) = rh_desikan_label(rh_desikan_label==i) - 1;
        else
            lh_desikan_label(lh_desikan_label==i) = lh_desikan_label(lh_desikan_label==i) - 2;
            rh_desikan_label(rh_desikan_label==i) = rh_desikan_label(rh_desikan_label==i) - 2;
        end
    end
    rh_desikan_label = rh_desikan_label + 34;
    rh_desikan_label(rh_desikan_label==34) = 0;
    all_desikan_label = [lh_desikan_label ; rh_desikan_label];
end

function vertice_val = roi2vertices(roi_val, labels)
    min_value = -500000;
    roi_num = size(roi_val, 1);
    vertice_val = ones(size(labels, 1), size(roi_val, 2)) * min_value;
    for i = 1:roi_num
        vertice_val(labels==i) = roi_val(i);
    end
end
    