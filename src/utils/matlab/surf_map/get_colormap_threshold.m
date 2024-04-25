function threshold=get_colormap_threshold(parameter)
    % First identify the whiskers for the given parameter,
    %   and set threshold to be the two end points of the two whiskers.
    % Then extend the two thresholds by 10% of the IQR.

    Q1 = quantile(parameter, 0.25);
    Q3 = quantile(parameter, 0.75);
    IQR = Q3 - Q1;
    whisker_up = Q3 + 1.5 * IQR;
    whisker_down = Q1 - 1.5 * IQR;

    whisker_up = max(parameter(parameter < whisker_up));
    whisker_down = min(parameter(parameter > whisker_down));

    threshold = [whisker_down, whisker_up];
    threshold(1) = threshold(1) - 0.1 * IQR;
    threshold(2) = threshold(2) + 0.1 * IQR;
end