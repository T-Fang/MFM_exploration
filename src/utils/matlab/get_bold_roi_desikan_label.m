function [bold_roi] = get_bold_roi_desikan_label(bold_raw, lh_label, rh_label)
% bold_raw: (64984, time_points)
% lh_label: (32492, 1)
% rh_label: (32492, 1)

bold_roi = NaN(68, size(bold_raw, 2));
for i = 1:36
   if i == 1 || i == 5
       continue
   elseif i < 5
       lh_pos = i - 1;
       rh_pos = i + 33;
       bold_roi(lh_pos, :) = CBIG_nanmean(bold_raw(lh_label == i, :), 1);
       bold_roi(rh_pos, :) = CBIG_nanmean(bold_raw(rh_label == i, :), 1);
   else
       lh_pos = i - 2;
       rh_pos = i + 32;
       bold_roi(lh_pos, :) = CBIG_nanmean(bold_raw(lh_label == i, :), 1);
       bold_roi(rh_pos, :) = CBIG_nanmean(bold_raw(rh_label == i, :), 1);
   end
end

end