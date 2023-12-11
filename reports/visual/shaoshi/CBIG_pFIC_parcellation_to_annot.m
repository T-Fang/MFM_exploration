function CBIG_pFIC_parcellation_to_annot(lh_label, rh_label, lh_filename, rh_filename, colortable)
    %%% lh_label and rh_label are vectors containing the left_hemisphere and right_hemisphere
    %%% code assumes lh_label and rh_label are 1 x N
    %%% we write into lh_filename and rh_filename
    %%%!! Colortable is either a matrix containing RGB numbers (e.g. [5 5 10; 255 0 0; 0 255 0; 0 0 255]) OR
    %%%!! the name of a colormap (e.g.'jet' or 'hot')
    % Written by CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    
    if(size(lh_label, 1) ~= 1)
        lh_label = lh_label';
    end
    
    if(size(rh_label, 1) ~= 1)
        rh_label = rh_label';
    end

    if(size(lh_label, 2) ~= size(rh_label, 2))
        error('size of lh_label not equal rh_label');
    end
    
    if(~exist('colortable', 'var'))
        colortable = 'jet'; 
    end
    
    label = [lh_label rh_label];
    vertices=0:(length(lh_label)-1);
    
    ct.numEntries = double(max(label)+1); %assume there is a 0 label, which is considered background/medial wall
    
    if(ischar(colortable))
        xcolormap = int32([30 10 30; feval(colortable, ct.numEntries-1)*255]);
        ct.orig_tab = colortable;
    else
        if(size(colortable, 2) ~= 3)
            error('If input colortable is a matrix, it should be a list of RGB values, thus there should be 3 columns.');
        end
        xcolormap = int32(colortable); 
        ct.orig_tab = 'manual';
    end
    
    ct.table = [xcolormap, zeros([ct.numEntries 1],'int32'), xcolormap(:,1) + xcolormap(:, 2)*2^8 + xcolormap(:, 3)*2^16]; %! in annot files, medial wall is annotated as 1
    
    new_lh_label = lh_label;
    new_rh_label = rh_label;
    
    for i=0:(ct.numEntries-1)
        if(i == 0)
            ct.struct_names{i+1} = 'unknown';
        else
            ct.struct_names{i+1}= ['parcel' num2str(i)];
        end
        new_lh_label(lh_label == i) = ct.table(i+1, 5);  
        new_rh_label(rh_label == i) = ct.table(i+1, 5); 
    end

    write_annotation(lh_filename, vertices, new_lh_label, ct)
    write_annotation(rh_filename, vertices, new_rh_label, ct)
end
