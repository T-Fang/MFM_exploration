function compare_FC(sim_FC_path,emp_FC_path, save_fig_path)

FC_test = csvread(emp_FC_path);
FC = csvread(sim_FC_path);

close all
hold on
plot(FC_test(triu(true(68), 1)), FC(triu(true(68), 1)), '.', 'Color', [105, 105, 105]/255, 'MarkerSize', 8)
xlim([-0.2, 1])
ylim([-0.2, 1])
set(gca, 'LineWidth', 2)
set(gca,'Fontsize', 10,'TickDir','out','FontWeight','bold')
set(gca,'LineWidth',2)
set(gca,'box','off')
set(gca, 'ytick', [0, 0.5, 1])
set(gca, 'xtick', [0, 0.5, 1])
xlabel('empirical FC', 'FontSize', 24)
ylabel('simulated FC', 'FontSize', 24)
coefficients = polyfit(FC_test(triu(true(68), 1)), FC(triu(true(68), 1)), 1);
xFit = linspace(-0.2, 1, 1000);
yFit = polyval(coefficients , xFit);
plot(xFit, yFit, 'r-', 'LineWidth', 3); % Plot fitted line.

% Calculate and display correlation coefficient
corr_coef = corrcoef(FC_test(triu(true(68), 1)), FC(triu(true(68), 1)));
text(0.7, 0.0, ['r=', num2str(round(corr_coef(1,2), 4))], 'FontSize', 20)

hold off
% print(save_fig_path, '-dsvg', '-r0')
print(save_fig_path, '-dpng', '-r100')
end

