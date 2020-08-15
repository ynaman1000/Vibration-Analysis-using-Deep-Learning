%% Defining Constants
% All physical constants are in SI units.
no_ns = 350;
n_min = 3+0.01;
n_max = 10-0.01;
order_arr = reshape(linspace(n_min, n_max, no_ns), [no_ns, 1]);      % exponent in area and moment of inertia; I1*eta^(n+2), A1*eta^n 
no_etas = 15000;     % number of data points to be generated
eta_min = 7e-2;
eta_max = 0.56;
etas = reshape(linspace(eta_min, eta_max, no_etas), [1, no_etas]);   % fraction of truncated portion, keep less than (0.6047 for n=0, 0.6018 for n=1, 0.5989 for n=2)
no_of_modes = 3;    % number of modes
beta_max = 20;      % max value of beta
dim_output = 6;   % dimension of output of training data to be generated, if this changed, then also change line 21 accordingly
train_output_data = zeros(no_etas, dim_output);
% csvwrite('train_data.csv', '');     %creating empty file
%%

% delete(gcp('nocreate'));
% parpool([4,20]);

tic;
for j = 3:no_ns
    n = order_arr(j, 1);
    %% Generating Training Data
    parfor k = 1:no_etas
        data = cell2mat(main_fun(n, etas(1, k), beta_max, no_of_modes));
        train_output_data(k, :) = [data(:, 1).' data(2, 2) data(3, 2) data(3, 3)];
    end
    %%

    %% Appending Training Data to file
    dlmwrite('train_data.csv', [n*ones(no_etas, 1) etas(1, :).' train_output_data], 'delimiter', ',', '-append');
    j
end
toc;

% delete(gcp('nocreate'));