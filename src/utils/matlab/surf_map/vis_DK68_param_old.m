function vis_DK68_param_old(parameter, save_fig_path, H)
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
h_fig = DrawSurfaceMaps_fslr(vertice_para(1:32492), vertice_para(32493:end), 'fs_LR_32k', 'very_inflated', ...
                               min_thres - 0.01 * (max_thres - min_thres), max_thres, H);
% h_fig = DrawSurfaceMaps_fslr(vertice_para(1:32492), vertice_para(32493:end), 'fs_LR_32k', 'very_inflated', ...
%                                -14.5e-04, -4e-04, H);
saveas(h_fig, save_fig_path);
close

end


function all_desikan_label = desikan_label_subcortical_removal()
% Subcortical will be set to 0.
    lh_desikan_label = load('/home/ftian/storage/projects/MFM_exploration/src/utils/matlab/labels/lh_desikan_label.mat');
    lh_desikan_label = lh_desikan_label.lh_desikan_fslr_32k_label;
    rh_desikan_label = load('/home/ftian/storage/projects/MFM_exploration/src/utils/matlab/labels/rh_desikan_label.mat');
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


function h = DrawSurfaceMaps_fslr(lh_data, rh_data, mesh_name, surf_type, min_thresh,max_thresh,colors)

    % h = DrawSurfaceMaps_fslr(lh_data, rh_data, mesh_name, surf_type, min_thresh,max_thresh,colors)
    %
    % This function a given labels (parcellation labels, activation patterns...)
    % in fslr to a Matlab figures of surfaces.
    %
    % Input:
    %      - lh_data, rh_data:
    %        surface labels on the left and right hemisphere respectively in
    %        fslr. Each variable is a Nx1 or 1xN vector, where N corresponds to the
    %        mesh defined in mesh_name.
    %        N = 163,842 for mesh_name = 'fs_LR_164k'
    %        N = 32,492 for mesh_name = 'fs_LR_32k'
    %
    %      - mesh_name:
    %        fslr mesh that the label data will be projected onto.
    %        fslr = 'fs_LR_32k' or 'fs_LR_164k'
    %      - surf_type:
    %        surface template. Options are 'inflated', 'very_inflated',
    %        'midthickness_orig', 'white_orig', 'pial_orig', 'sphere'.
    %      - min_thresh, max_thresh (optional):
    %        minimum and maximum threshold for the colorscale of the projected
    %        values.
    %      - colors (optional):
    %        custom colorscale of the projected values
    %
    % Output:
    %      - h:
    %        handle of the figure
    %
    % Example:
    % - DrawSurfaceMaps_fslr(lh_proj_32K,rh_proj_32K, 'fs_LR_32k', 'inflated', 1e-05, 5e-5);
    %   Draw label data onto fslr_32k inflated surface mesh with a colorscale
    %   between 1e-05 and 5e-05.
    %
    % Written by CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

    % This function does not need vector check because the function itself
    % contains checking statement.

    warning('off', 'MATLAB:warn_r14_stucture_assignment');

    pos = [0.020, 0.510, 0.325, 0.470;...
        0.355, 0.510, 0.325, 0.470;...
        0.720, 0.760, 0.240, 0.230;...
        0.720, 0.510, 0.240, 0.230;...
        0.020, 0.020, 0.325, 0.470;...
        0.355, 0.020, 0.325, 0.470;...
        0.720, 0.260, 0.240, 0.230;...
        0.720, 0.010, 0.240, 0.230];

    h = figure; gpos = get(h, 'Position');
    gpos(1) = 0; gpos(2) = 0; gpos(3) = 1200; gpos(4) = 600; set(h, 'Position', gpos);

    if(exist('colors', 'var'))
        m = colors/max(colors(:));
        colormap(m);
    else
        m = jet;
        colormap(m);
    end

    %Add threshold if not specified
    if(~exist('min_thresh', 'var'))
        min_thresh=min([min(lh_data) min(rh_data)]);
        max_thresh=max([max(lh_data) max(rh_data)]);
    end


    for hemis = {'lh','rh'}
        
        hemi = hemis{1};
        mesh= CBIG_read_fslr_surface(hemi, mesh_name, surf_type, 'medialwall.annot');
        non_cortex = find(mesh.MARS_label == 1);
        
        if(strcmp(hemi, 'lh'))
            data = single(lh_data);
        elseif(strcmp(hemi, 'rh'))
            data = single(rh_data);
        end
        
        % convert to row vector
        if(size(data, 1) ~= 1)
            data = data';
        end
        
        %threshold
        data(data < min_thresh) = min_thresh;
        data(data > max_thresh) = max_thresh;
        data(non_cortex(1)) = min_thresh;
        data(non_cortex(2)) = max_thresh;
        
        
        % draw
        if(strcmp(hemi, 'lh'))
            subplot('Position', pos(1, :));
            s = TrisurfMeshData(mesh, data);
            shading interp;
            ncd = revert_shading_interp_behaviour(s, m);
            s.CData = ncd;
            view(-90, 0);
            axis off;
            
            subplot('Position', pos(2, :));
            s = TrisurfMeshData(mesh, data);
            shading interp;
            ncd = revert_shading_interp_behaviour(s, m);
            s.CData = ncd;
            view(90, 0);
            axis off;
            
            subplot('Position', pos(3, :));
            s = TrisurfMeshData(mesh, data);
            shading interp;
            ncd = revert_shading_interp_behaviour(s, m);
            s.CData = ncd;
            view(90, 90);
            axis off;
            
            subplot('Position', pos(8, :));
            s = TrisurfMeshData(mesh, data);
            shading interp;
            ncd = revert_shading_interp_behaviour(s, m);
            s.CData = ncd;
            view(90, -90);
            axis off;
            
        else
            
            subplot('Position', pos(5, :));
            s = TrisurfMeshData(mesh, data);
            shading interp;
            ncd = revert_shading_interp_behaviour(s, m);
            s.CData = ncd;
            view(90, 0);
            axis off;
            
            subplot('Position', pos(6, :));
            s = TrisurfMeshData(mesh, data);
            shading interp;
            ncd = revert_shading_interp_behaviour(s, m);
            s.CData = ncd;
            view(-90, 0);
            axis off;
            
            subplot('Position', pos(4, :));
            s = TrisurfMeshData(mesh, data);
            shading interp;
            ncd = revert_shading_interp_behaviour(s, m);
            s.CData = ncd;
            view(90, 90);
            axis off;
            
            subplot('Position', pos(7, :));
            s = TrisurfMeshData(mesh, data);
            shading interp;
            ncd = revert_shading_interp_behaviour(s, m);
            s.CData = ncd;
            view(90, -90);
            axis off;
        end
    end

    if(exist('min_thresh', 'var'))
        cbax = axes('Position', [0.29 0.5 0.1 0.02], 'visible', 'off');
        caxis(cbax, [min_thresh, max_thresh]);
        cb = colorbar('peer', cbax, 'horiz', 'Position', [0.29 0.5 0.1 0.02]);
        cb.FontSize = 24; % Set the font size to your desired value
    end

end


