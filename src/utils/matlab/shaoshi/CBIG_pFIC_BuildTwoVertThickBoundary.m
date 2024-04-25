function [lh_up_labels,rh_up_labels] = CBIG_pFIC_BuildTwoVertThickBoundary(lh_avg_mesh,rh_avg_mesh,lh_full_label,rh_full_label)

% this function defines the boundary for a given parcellation
% by default, no parcellation comes with pre-defined boundaries
% this function declares a vertex as non-boundary only if its neighbored by vertices with the same label
% Written by CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

if(size(lh_full_label, 1) == 1)
   lh_full_label = lh_full_label';
end

if(size(rh_full_label, 1) == 1)
   rh_full_label = rh_full_label';
end

lh_neigh_labels=lh_avg_mesh.vertexNbors;
rh_neigh_labels=rh_avg_mesh.vertexNbors;
lh_neigh_labels(lh_neigh_labels~=0)=lh_full_label(lh_neigh_labels(lh_neigh_labels~=0));
rh_neigh_labels(rh_neigh_labels~=0)=rh_full_label(rh_neigh_labels(rh_neigh_labels~=0));

lh_full_labels=[lh_full_label';lh_neigh_labels];
rh_full_labels=[rh_full_label';rh_neigh_labels];
lh_temp=bsxfun(@minus,lh_full_labels,lh_full_labels(1,:));
rh_temp=bsxfun(@minus,rh_full_labels,rh_full_labels(1,:));
lh_temp(lh_full_labels==0)=0;
rh_temp(rh_full_labels==0)=0;
lh_up_labels=lh_full_label;
rh_up_labels=rh_full_label;
lh_up_labels(sum(lh_temp)~=0)=0;
rh_up_labels(sum(rh_temp)~=0)=0;
