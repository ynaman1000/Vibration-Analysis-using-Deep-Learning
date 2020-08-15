%% Defining Constants
% All physical constants are in SI units.
n = 2;      % exponent in area and moment of inertia; I1*eta^(n+2), A1*eta^n 
eta0 = linspace(1e-8, 0.5989, 10000);   % fraction of truncated portion, keep less than (0.6047 for n=0, 0.6018 for n=1, 0.5989 for n=2)
no_of_modes = 3;    % number of modes
beta_max = 20;      % max value of beta
dim_data_out = 6;   % dimension of output of training data to be generated, if this changed, then also change line 16 accordingly
%%

tic;
%% Generating Training Data
no_pts = size(eta0, 2);
train_data = zeros(no_pts, dim_data_out);
parfor i = 1:no_pts
    data = cell2mat(main_fun(n, eta0(i), beta_max, no_of_modes));
    train_data(i, :) = [data(:, 1).' data(2, 2) data(3, 2) data(3, 3)];
end
%%
toc;

%% Saving Training Data
train_data = [eta0.' train_data];
csvwrite('train_data.csv', train_data);
