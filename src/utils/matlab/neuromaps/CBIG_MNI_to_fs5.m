function fs5_data = CBIG_MNI_to_fs5(mni_file_path, fs5_file_path)
%CBIG_MNI_TO_FS5 convert the data stored in the mni file to the fs7 space
% and save to fs5_file_path
    [lh_fs7_data, rh_fs7_data] = CBIG_MNI_to_fs7(mni_file_path);
    lh_fs5_data = downsample_fs7('lh', lh_fs7_data, 'fsaverage5');
    rh_fs5_data = downsample_fs7('rh', rh_fs7_data, 'fsaverage5');
    fs5_data = [lh_fs5_data, rh_fs5_data];

    % save to fs5_file_path as a csv file if the argument exist
    if nargin == 2
        csvwrite(fs5_file_path, fs5_data);
    end
end

