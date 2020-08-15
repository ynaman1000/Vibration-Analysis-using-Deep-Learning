%% Defining Constants
% All physical constants are in SI units.
n = 2;      % exponent in area and moment of inertia; I1*eta^(n+2), A1*eta^n 
L1 = 1;     % length of cone
L0 = 0.1;   % length of truncated portion, keep less than 1.1978
R1o = 0.01;  % outer radius at base
R1i = 0.009; % inner radius at base
E = 200e6;      % Young's modulus of steel
Rho = 7900;     % Density of steel

eta0 = L0/L1;
eta1 = 1;
eta_int = [eta0 eta1];
beta_int = [0.1, 20];
%


% Finding betas
det_A = @(beta) det(A(beta, n, eta0, eta1));
f1 = chebfun(det_A , beta_int, 'splitting', 'on', 'minSamples', 33);
rf1 = roots(f1);
beta_int(1, 2) = ceil(rf1(3));
f1 = chebfun(det_A , beta_int, 'splitting', 'on', 'minSamples', 33);
rf1 = roots(f1);
betas = rf1(:, 1);
err = f1(betas)
%%


%% Finding Natural Frequencies, Nodes and Plotting Mode shapes
nat_freqs = omega(betas, L1, R1o, R1i, E, Rho) / (2*pi);

no_modes = size(betas, 1);
nodes = zeros(no_modes, no_modes);
c = zeros(no_modes, 4);
for i = 1:no_modes
    c(i, :) = null(A(betas(i, 1), n, eta0, eta1)).';
end
for i = 1:no_modes
    mode_shape_fun = @(eta) W(eta, n, L1, betas(i, 1), c(i, :));
    f2 = chebfun(mode_shape_fun, eta_int, 'splitting', 'on');
    rf2 = roots(f2);
    nodes(i, 1:size(rf2, 1)) = rf2.';
    try
        assert(nodes(i, i) ~= 0);
    catch
        nodes(i, i) = 1;
    end
end
nodes
fplot(mode_shape_fun, eta_int);
%%


% fzero(d, 3);