function [lh_proj_data, rh_proj_data] = CBIG_MNI_to_fs7(mni_file_path, fs7_file_path)
%CBIG_MNI_TO_FS7 convert the data stored in the mni file to the fs7 space
% and save to fs7_file_path

    data = MRIread(mni_file_path);
    [lh_proj_data, rh_proj_data] = CBIG_ProjectMNI2fsaverage_Ants(data, 'fsaverage');
    fs7_data = [lh_proj_data, rh_proj_data];

    % save to fs7_file_path as a csv file if the argument exist
    if nargin == 2
        csvwrite(fs7_file_path, fs7_data);

end

