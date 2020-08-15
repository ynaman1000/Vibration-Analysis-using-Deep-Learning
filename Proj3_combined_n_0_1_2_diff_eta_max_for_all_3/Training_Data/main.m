%% Defining Constants
% All physical constants are in SI units.
order_arr = [0; 1; 2];      % exponent in area and moment of inertia; I1*eta^(n+2), A1*eta^n 
no_pts = 10000;     % number of data points to be generated
etas = [linspace(1e-8, 0.6047, no_pts); ...   % fraction of truncated portion, keep less than (0.6047 for n=0, 0.6018 for n=1, 0.5989 for n=2)
        linspace(1e-8, 0.6018, no_pts); ...
        linspace(1e-8, 0.5989, no_pts)];
no_of_modes = 3;    % number of modes
beta_max = 20;      % max value of beta
dim_output = 6;   % dimension of output of training data to be generated, if this changed, then also change line 21 accordingly
train_output_data = zeros(no_pts, dim_output);
csvwrite('train_data.csv', '');     %creating empty file
%%

tic;
for j = 1:3
    n = order_arr(j, 1);
    %% Generating Training Data
    parfor i = 1:no_pts
        data = cell2mat(main_fun(n, etas(j, i), beta_max, no_of_modes));
        train_output_data(i, :) = [data(:, 1).' data(2, 2) data(3, 2) data(3, 3)];
    end
    %%

    %% Appending Training Data to file
    dlmwrite('train_data.csv', [n*ones(no_pts, 1) etas(j, :).' train_output_data], 'delimiter', ',', '-append');
end
toc;