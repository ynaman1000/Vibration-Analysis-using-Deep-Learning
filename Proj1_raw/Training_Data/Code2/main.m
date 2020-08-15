%% Defining Constants
% All physical constants are in SI units.
n = 2;      % exponent in area and moment of inertia; I1*eta^(n+2), A1*eta^n 
eta0 = linspace(1e-8, .5989, 10000);   % fraction of truncated portion, keep less than 0.5989
no_of_modes = 3;
%%

tic;
%% Generating Training Data
no_pts = size(eta0, 2);
train_data = zeros(no_pts, 6);
parfor i = 1:no_pts
    data = cell2mat(main_fun(n, eta0(i), no_of_modes));
    train_data(i, :) = [data(:, 1).' data(2, 2) data(3, 2) data(3, 3)];
end
%%
toc;

%% Saving Training Data
train_data = [eta0.' train_data];
csvwrite('train_data.csv', train_data);
