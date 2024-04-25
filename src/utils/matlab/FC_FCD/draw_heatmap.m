function draw_heatmap(heatmap_data_path,title,save_file_path)
% Draw the heatmap and save as a png file, using the pre-defined color map
% Input:
%  - heatmap_data_path: the path to the heatmap data csv file
%  - title: the title of the heatmap
%  - save_file_path: the path to save the heatmap image
%
% Written by Tian Fang under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

heatmap = csvread(heatmap_data_path);
load('FCD_color_table.mat', 'c3')
imagesc(heatmap);
colormap(c3)
set(gca, 'CLim', [0.1, 1])
set(gca, 'YTickLabel', [])
set(gca, 'XTickLabel', [])
set(gca, 'LineWidth', 2)
set(gca,'xtick',[])
set(gca,'ytick',[])
set(get(gca, 'title'), 'string', title)
% title(gca, title)

% print(save_file_path, '-dsvg', '-r0')
print(save_file_path, '-dpng', '-r100')

close all
end

