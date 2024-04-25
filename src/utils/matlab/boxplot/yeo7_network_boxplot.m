function yeo7_network_boxplot(ROI_stats, stats_name, save_fig_path)

% yeo7_network_boxplot()
% This function generates a box plot depicting the network pattern of the input statistic
% 
% Input:
%   - ROI_stats: a 68x1 vector, each of the value represents the statistic of a ROI
%   - save_fig_path: the path to save the figure
%   - H: an optional color map
% 
% Written by Tian Fang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

% insert the medial wall ROIs
ROI_stats_72 = [0; ROI_stats(1:3); 0; ROI_stats(4:34); 0; ROI_stats(35:37); 0; ROI_stats(38:68)];
network_assignment = CBIG_pFIC_ROI2network(ROI_stats_72);
% network_assignment: a 14114-by-2 matrix. The first column is the network index 
% from 1 to 7. The second column is the value of vertices corresponding to the network.
boxplot(network_assignment(:, 2), network_assignment(:, 1), 'Colors', 'k', 'Positions', 1:0.8:5.8, 'Widths', 0.5)
set(findobj(gcf,'tag','Outliers'), 'MarkerEdgeColor', 'black', 'MarkerSize', 5, 'Marker', '+')
set(findobj(gca,'type','line'),'linew',1.5)
set(gca,'Fontsize',10,'TickDir','out','FontWeight','bold')
set(gca,'LineWidth',2)
set(gca,'box','off')
% set(gca,'ytick', [0, 0.6, 1.2])
ylabel(stats_name)
set(gca,'xticklabel',{'Som', 'Vis', 'DA', 'VA', 'Lim', 'Control', 'Default'})
set(gcf,'Position',[800,100,1000,700])
% print(save_fig_path, '-dsvg', '-r0')
fig = gcf; % get the current figure
saveas(fig, save_fig_path); % save the figure
close all

end
