function [TC_DK68] = convert_TC_fslr2DK68(bold_raw, lh_label, rh_label)
    % bold_raw: (64984, time_points)
    % lh_label: (32492, 1)
    
    time_points = size(bold_raw, 2);
    
    % Wrong code
    % TC_DK68 = NaN(68, time_points);
    % for i = 1:36
    %    if i == 1 || i == 5
    %        continue
    %    elseif i < 5
    %        lh_pos = i - 1;
    %        rh_pos = i + 33;
    %        bold_roi(lh_pos, :) = CBIG_nanmean(bold_raw(lh_label == i, :), 1);
    %        bold_roi(rh_pos, :) = CBIG_nanmean(bold_raw(rh_label == i, :), 1);
    %    else
    %        lh_pos = i - 2;
    %        rh_pos = i + 32;
    %        bold_roi(lh_pos, :) = CBIG_nanmean(bold_raw(lh_label == i, :), 1);
    %        bold_roi(rh_pos, :) = CBIG_nanmean(bold_raw(rh_label == i, :), 1);
    %    end
    % end
    
    surface_lh = bold_raw(1:32492, :);
    surface_rh = bold_raw(32493:64984, :);
    
    TC_lh = NaN(36, time_points);
    TC_rh = NaN(36, time_points);
    
    for i = 1:36
        TC_lh(i, :) = CBIG_nanmean(surface_lh(lh_label == i, :), 1);
        TC_rh(i, :) = CBIG_nanmean(surface_rh(rh_label == i, :), 1); 
    end
    
    TC = [TC_lh; TC_rh];
    
    DK72_mask = true(72, 1);
    DK72_mask([1 5 37 41]) = 0;
    
    TC_DK68 = TC(DK72_mask, :);
    
    
    end