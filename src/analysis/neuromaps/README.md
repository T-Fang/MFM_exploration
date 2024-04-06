# Apply PCA on Neuro Maps to Find out the Maps that encapsulate the most information

### Surface Mesh to the Desikan Killiany Atlas

Use MATLAB function to read the label mapping template from surface meshes to the Desikan parcellation.

- fsLR: `fsLR_lh_label = CBIG_read_annotation('fs_LR_164k_lh.aparc.annot');`
- fsaverage: `lh_label = CBIG_ReadNCAvgMesh('lh', 'fsaverage5', 'inflated', 'aparc.annot'); lh_label = lh_label.MARS_label; % 1 to 36`

> **Note**: the templates map to the Desikan Atlas that has 72 ROI. For our purpose, we will ignore medial wall ROIs 'unknown' and 'corpuscallosum'. Correspondingly, the ROIs 1, 5, 37, 41 (1-based) are ignored.
