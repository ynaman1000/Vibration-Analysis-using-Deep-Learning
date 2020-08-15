function C = main_fun(n, e, b, no_modes)
    % n- integer, exponent in area and moment of inertia
    % e- float, truncated fraction
    % b- number, max value of beta
    % no_modes- integer, number of modes
    %% Defining Intervals
    eta0 = e;
    eta1 = 1;
    eta_int = [eta0 eta1];
    beta_int = [1, b];
    %%


    %% Finding betas
    det_A = @(beta) det(A(beta, n, eta0, eta1));
    f1 = chebfun(det_A , beta_int, 'splitting', 'on', 'minSamples', 33);
    rf1 = roots(f1);
    beta_int(1, 2) = ceil(rf1(no_modes));
    f1 = chebfun(det_A , beta_int, 'splitting', 'on', 'minSamples', 33);
    rf1 = roots(f1);
    betas = rf1(1:no_modes, 1);
    %%


    %% Finding Nodes
    nodes = zeros(no_modes, no_modes);
    c = zeros(no_modes, 4);
    for i = 1:no_modes
        temp = null(A(betas(i, 1), n, eta0, eta1));
        if isempty(temp)
            temp = null(A(betas(i, 1), n, eta0, eta1));
        else
            c(i, :) = temp.';
        end
%         c(i, :) = null(A(betas(i, 1), n, eta0, eta1)).';
    end
    for i = 1:no_modes
        mode_shape_fun = @(eta) W(eta, n, betas(i, 1), c(i, :));
        f2 = chebfun(mode_shape_fun, eta_int, 'splitting', 'on', 'minSamples', 33);
        rf2 = roots(f2);
        try
            assert(rf2(i, 1) ~= 0);
        catch
            rf2(i, 1) = 1;
        end
        rf2 = rf2(1:i, 1);
        nodes(i, 1:i) = rf2.';
    end
    %%

    C = {betas, nodes};
end