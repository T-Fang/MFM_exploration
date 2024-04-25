function out_data = downsample_fs7(hemi, input, target_mesh)
% downsample from 'fsaverage7' to either 'fsaverage5' or 'fsaverage5', which
%  can be specified by the target_mesh argument

% Read the csv file if the input is a string, otherwise the input should be a matrix
  if ischar(input)
      input = csvread(input);
  end
  input_data = input;

  % assert that the size of the input is correct
  assert(size(input_data, 2) == 163842, 'Input data should have 163842 vertices');

  input_mesh = CBIG_ReadNCAvgMesh(hemi, 'fsaverage', 'sphere', 'cortex');
  num_vertices = size(input_mesh.vertices, 2);

  output_mesh = CBIG_ReadNCAvgMesh(hemi, target_mesh, 'sphere', 'cortex');
  num_out_vertices = size(output_mesh.vertices, 2);

  if(num_out_vertices == num_vertices)
      out_data = input_data;
  else
      % Map out Closest Vertex
      index = MARS_findNV_kdTree(input_mesh.vertices, output_mesh.vertices);

      % Downsample data
      out_data = zeros(size(input_data, 1),  num_out_vertices);
      for i = 1:num_out_vertices
          if(output_mesh.MARS_label(i) == 2) %if output vertex is cortex
              out_data(:, i) = mean(input_data(:, index == i & input_mesh.MARS_label == 2), 2);
          else
              out_data(:, i) = input_data(:, i);
          end
      end
  end
end

