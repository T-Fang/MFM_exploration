function H = Num2Color(Wvector,colormap_name)

    %-------------------------------------------------------------------------
    % H = Num2Color(Wvector,colormap_name)
    %
    % This function transfers vector value into a color map
    %
    % Input: 
    %      - Wvector:       parameter value vector {Parameters x 1}
    %      - colormap_name: color map name in Matlab, i.e. 'cool'
    % Output:
    %      - H:  a new colormap paired with the values in parameter vector
    %
    % Written by Peng Wang under MIT license: 
    % https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    %--------------------------------------------------------------------------
    
    figure
    colormap(colormap_name);
    C = colormap;
    L = size(C,1);
    [~, Windex] = sort(Wvector);
    H_max = Wvector(Windex(end));
    H_min = Wvector(Windex(1));
    Ws = round(interp1(linspace(H_min,H_max,L),1:L,Wvector));
    H = reshape(C(Ws,:),[size(Ws) 3]);
    H = squeeze(H(:, 1, :)).*255;
    close
    
end