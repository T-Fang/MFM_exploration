function CBIG_DrawSurfaceMapsWithBoundary_mod(lh_data, rh_data, ...
    lh_labels, rh_labels, mesh_name, surf_type, min_thresh, max_thresh, colors, parameter, fig_title)

    % CBIG_DrawSurfaceMapsWithBoundary(lh_data, rh_data, ...
    %     lh_labels, rh_labels, mesh_name, surf_type, min_thresh, max_thresh, colors)
    %
    % This function visualizes lh_data and rh_data with boundary defined 
    % by a parcellation lh_labels, rh_labels in freesurfer space Threshold 
    % can be defined by 
    % min_thresh and max_thresh.
    %
    % Input:
    %      -lh_data, rh_data: 
    %       data of left/right hemisphere. Nx1 or 1xN vector for each, 
    %       N = # of vertices in mesh_name.
    %      
    %      -lh_labels, rh_labels:
    %       parcellation of data of left/right hemisphere. Nx1 or 1xN vector for each, 
    %       N = # of vertices in mesh_name.
    %
    %      -mesh_name:
    %       Freesurfer mesh structure. For example, 'fsaverage5'.
    %
    %      -surf_type:
    %       Freesurfer surface template. Can be 'inflated', 'sphere', or
    %       'white'.
    %
    %      -min_thresh, max_thresh:
    %       min and max threshold for lh_data and rh_data. If they are not
    %       given, then there is no threshold.
    %
    %      -colors:
    %       color map for visualizetion. A Lx3 matrix, where L is the number of
    %       different colors for lh_data and rh_data. Each row is the R, G, B
    %       value. If colors is not given, visualization color will be defined
    %       by default Matlab colormap.
    %
    % Example:
    % CBIG_DrawSurfaceMapsWithBoundary(lh_data, rh_data, lh_labels,rh_labels,'fsaverage5','inflated');
    %
    % Written by CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

    % This function does not need vector check because the function itself
    % contains checking statement.

    warning('off', 'MATLAB:warn_r14_stucture_assignment');

    if(~exist('mesh_name', 'var'))
        mesh_name = 'fsaverage'; 
    end

    if(~exist('surf_type', 'var'))
        surf_type = 'inflated'; 
    end

    pos = [0.020, 0.510, 0.415, 0.470;...
        0.455, 0.510, 0.415, 0.470;...
        0.720, 0.760, 0.240, 0.230;...
        0.720, 0.510, 0.240, 0.230;...
        0.020, 0.020, 0.415, 0.470;...
        0.455, 0.020, 0.415, 0.470;...
        0.720, 0.260, 0.240, 0.230;...
        0.720, 0.010, 0.240, 0.230];

    h = figure; gpos = get(h, 'Position');
    gpos(1) = 100; gpos(2) = 100; gpos(3) = 800; gpos(4) = 600; set(h, 'Position', gpos);

    if(exist('colors', 'var'))
        m = colors/max(colors(:));
        colormap(m);
    else
        m = jet;
        colormap(m);
    end

    for hemis = {'lh' 'rh'}
        
        hemi = hemis{1};
        mesh = CBIG_ReadNCAvgMesh(hemi, mesh_name, surf_type, 'cortex');
        non_cortex = mesh.MARS_label == 1;  
        
        if(strcmp(hemi, 'lh'))
            data   = lh_data;
            labels = lh_labels;
        elseif(strcmp(hemi, 'rh'))
            data   = rh_data;
            labels = rh_labels;
        end

        % convert to row vector
        if(size(data, 1) ~= 1)
            data = data';  
        end
        
        if(size(labels, 1) ~= 1)
            labels = labels';  
        end
        
        % resample
        if(size(mesh.vertices, 2) ~= length(data)) % need to resample!
            if(length(data) == 10242)
                from_mesh = CBIG_ReadNCAvgMesh(hemi, 'fsaverage5', 'sphere', 'cortex');
                target_mesh = CBIG_ReadNCAvgMesh(hemi, mesh_name, 'sphere', 'cortex');
                data   = MARS_linearInterpolate(target_mesh.vertices, from_mesh, data);
                labels = MARS_NNInterpolate(target_mesh.vertices, from_mesh, labels);
            else
                error(['Not handling ' num2str(length(data)) ' vertices']);
            end
        end
        
        % threshold
        if(exist('min_thresh', 'var'))
            data(data < min_thresh) = min_thresh;
            data(data > max_thresh) = max_thresh;
            %data(non_cortex(1)) = min_thresh;
            data(non_cortex) = min_thresh;
        end
        
        % compute boundary
        BoundaryVec = zeros(length(labels), 1);
        maxNeighbors = size(mesh.vertexNbors, 1);
        for i = 1:length(labels)
            label_vertex = int32(labels(i));
            
            for k = 1:maxNeighbors
                v_neighbor = mesh.vertexNbors(k, i);
                if(v_neighbor ~= 0 && int32(labels(v_neighbor)) ~= label_vertex)
                    BoundaryVec(i) = 1;
                end
            end
        end
        data(BoundaryVec == 1) = max_thresh;
        
        % draw
        if(strcmp(hemi, 'lh'))
            subplot('Position', pos(1, :)); 
            s = TrisurfMeshData(mesh, data);
            %shading interp;
            %ncd = revert_shading_interp_behaviour(s, m);
            %s.CData = ncd;
            shading('FLAT')
            view(-90, 0);
            axis off; 
            title(fig_title, 'Units', 'normalized', 'Position', [1, 1, 0], 'FontSize', 16)
            
            subplot('Position', pos(2, :));
            s = TrisurfMeshData(mesh, data);
            %shading interp;
            %ncd = revert_shading_interp_behaviour(s, m);
            %s.CData = ncd;
            shading('FLAT')
            view(90, 0);
            axis off;
            
            %subplot('Position', pos(3, :));
            %s = TrisurfMeshData(mesh, data);
            %shading interp;
            %ncd = revert_shading_interp_behaviour(s, m);
            %s.CData = ncd;
            %view(90, 90);
            %axis off;

            %subplot('Position', pos(8, :)); 
            %s = TrisurfMeshData(mesh, data);
            %shading interp;
            %ncd = revert_shading_interp_behaviour(s, m);
            %s.CData = ncd;
            %view(90, -90);
            %axis off;  
            
        else

            subplot('Position', pos(5, :));
            s = TrisurfMeshData(mesh, data);
            %shading interp;
            %ncd = revert_shading_interp_behaviour(s, m);
            %s.CData = ncd;
            shading('FLAT')
            view(90, 0);
            axis off;

            subplot('Position', pos(6, :));
            s = TrisurfMeshData(mesh, data);
            %shading interp;
            %ncd = revert_shading_interp_behaviour(s, m);
            %s.CData = ncd;
            shading('FLAT')
            view(-90, 0);
            axis off;

            %subplot('Position', pos(4, :));
            %s = TrisurfMeshData(mesh, data);
            %shading interp;
            %ncd = revert_shading_interp_behaviour(s, m);
            %s.CData = ncd;
            %view(90, 90);
            %axis off;

            %subplot('Position', pos(7, :));
            %s = TrisurfMeshData(mesh, data);
            %shading interp;
            %ncd = revert_shading_interp_behaviour(s, m);
            %s.CData = ncd;
            %view(90, -90);
            %axis off;
        end
    end

    %if(exist('min_thresh', 'var'))
    %    cbax = axes('Position', [0.29 0.5 0.1 0.02], 'visible', 'off');
    %    data = [lh_data; rh_data]; 
    %    data(data < min_thresh) = min_thresh; 
    %    data(data > max_thresh) = max_thresh;
    %    caxis(cbax, [min(min(data)), max(max(data))]);
    %    colorbar('peer', cbax, 'horiz', 'Position', [0.29 0.5 0.1 0.02]);
    %end  

    ax1 = axes;
    ax1.Visible = 'off';
    ax1.XTick = [];
    ax1.YTick = [];
    colormap(ax1, 'cool');
    set(gca, 'CLim', [min(parameter), max(parameter)])
    % colorbar(ax1, 'horiz', 'Position', [0.28 0.5 0.2 0.02], 'XTick', [min(parameter), max(parameter)],'XTickLabel', {num2str(min(parameter), '%0.2g'), num2str(max(parameter), '%0.2g')}); 
    cbar = colorbar(ax1, 'horiz', 'Position', [0.35 0.5 0.2 0.02], 'XTick', [min(parameter), max(parameter)],'XTickLabel', {num2str(min(parameter), '%0.2g'), num2str(max(parameter), '%0.2g')}); 
    cbar.FontSize = 28;
end