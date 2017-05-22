function [ U, V, info ] = gspartan(Grids_tr_var, L, lambda1, lambda2, lambda3, k )
%FORMULAJOINT Summary of this function goes here
%   Detailed explanation goes here

% This function is the main function of algorithm Formula.
% Input:
% -- Grids_tr: contains all the information of the (training) data: X is
% standardized and added the bias
% -- L: similarity between locations
% -- lambda1, lambda2, lambda3: parameters for regularizations
% -- k: number of hidden tasks
% --    lambda1 : sparse for V
% --    lambda2 : sparse for U
% --    lambda3 : model neighborhood reconstruction. 
%
% Output:
% -- U: DxK matrix -- basis for models
% -- V: KxN matrix -- task assignment matrix
rng(0);
n = length(Grids_tr_var);
X = Grids_tr_var{1}.X;
[~, d] = size(X);

% initialization U and V
% U = zeros(d,k);
% V = zeros(k,n);
%% single task for all data and use W as the init value for U
lambda= [0.001 0.01 0.1 1];
perform_mat = zeros(length(lambda), 1);
Ws = cell(length(lambda), 1);

X = [];
Y = [];
for i = 1:n
	X = [X; Grids_tr_var{i}.X];
	Y = [Y; Grids_tr_var{i}.Y];
end

for lidx = 1:length(lambda)
    W0= SingleModelLasso(X, Y, lambda(lidx));
%     W0 = ridge(Y, X, lambda(lidx));
    y_pred = X * W0;
    mse = sqrt(sum((y_pred - Y).^2)) / length(y_pred);
    perform_mat(lidx) =  mse;
    Ws{lidx} = W0;
end

% use the best single task learning weights for initialize W in multi-task
% learning
[~, bestLambdaIndex] = min(perform_mat);
% W_init = Ws{bestLambdaIndex};
W_init = repmat(Ws{bestLambdaIndex}, [1,n]);

% factorize W_init to get initial U and V
% U = rand(d,k);
% V = U\W_init;
% U = repmat(Ws{bestLambdaIndex}, [1,k]);
U = zeros(d,k);
U = U + rand(size(U));

%V = ones(k,n)/k;
V = zeros(k,n);
V = V + rand(size(V));

UV_vect0 = [U(:); V(:)];

% function value/gradient of the smooth part
smoothF    = @(UV_vector) smooth_part( UV_vector, Grids_tr_var, L, lambda3);
% non-negativen l1 norm proximal operator.
non_smooth = prox_nnl1_Vonly(lambda1, lambda2,  size(U, 1) * size(U, 2) + 1 );

sparsa_options = pnopt_optimset(...
    'display'   , 0    ,...
    'debug'     , 0    ,...
    'maxIter'   , 1000  ,...
    'ftol'      , 1e-5 ,...
    'optim_tol' , 1e-5 ,...
    'xtol'      , 1e-5 ...
    );
[UV_vect, ~,info] = pnopt_sparsa( smoothF, non_smooth, UV_vect0, sparsa_options );

% U=> d * k
U = reshape (UV_vect(1:d*k), [d , k]);
% V=> k * n (remaining)
V = reshape (UV_vect(d*k + 1:end), [k , n]);

end

function [f, g] = smooth_part(UV_vect, Grids_tr_var, L, lambda3)
n = length(Grids_tr_var);
[~, d] = size(Grids_tr_var{1}.X);

k = length(UV_vect)/(n + d);
% U=> d * k
U = reshape (UV_vect(1:d*k), [d , k]);
% V=> k * n (remaining)
V = reshape (UV_vect(d*k + 1:end), [k , n]);
Lap = eye(n) - L;
W = U*V;

% gradient w.r.t U
% A = V * L;
% gradientU = lambda3 * U * (A * A');
gradientU = lambda3 * U * V * Lap * V';
for i = 1:n
    %xi = X(i,:)';
    xi = Grids_tr_var{i}.X;
	Yi = Grids_tr_var{i}.Y;
	vi = V(:,i);
    tmp = - xi' * Yi* vi' + xi' * xi * U * vi * vi';
	gradientU = gradientU + tmp;
end

% gradient w.r.t V
gradientV = [];
for i = 1:n
	Xi = Grids_tr_var{i}.X;
	Yi = Grids_tr_var{i}.Y;
	X_tilde = Xi*U;
    gradientV = [gradientV, X_tilde'*Yi + X_tilde'*X_tilde*V(:,i)];
end

% gradientV = gradientV + lambda3 * (U' * U) * ((V * B) * B');
gradientV = gradientV + lambda3 * (U' * U) * V * Lap;
% function value
%f = 0.5 * sum((Y' - (sum(V.*X_tilde'))).^2) ...
%    + lambda3/2 * norm(U*A, 'fro')^2;
% f = lambda3/2 * norm(U*A, 'fro')^2;
f = lambda3/2 * trace(W * Lap * W');
for i = 1:n
	Xi = Grids_tr_var{i}.X;
	Yi = Grids_tr_var{i}.Y;
	wi = U * V(:,i);
	f = f + 0.5 * norm(Yi - Xi * wi)^2;
end

% gradient
g = [gradientU(:); gradientV(:)];

end


function op = prox_nnl1_Vonly( q , r,  V_startIdx)

%PROX_L1    L1 norm.
%    OP = PROX_L1( q ) implements the nonsmooth function
%        OP(X) = norm(q.*X,1).
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).
% Dual: proj_linf.m

% Update Feb 2011, allowing q to be a vector
% Update Mar 2012, allow stepsize to be a vector

if nargin == 0,
    q = 1;
elseif ~isnumeric( q ) || ~isreal( q ) ||  any( q < 0 ) || all(q==0) %|| numel( q ) ~= 1
    error( 'Argument must be positive.' );
end

op = tfocs_prox( @f, @prox_f , 'vector' ); % Allow vector stepsizes

    function v = f(x)
        %     v = norm( q(V_startIdx:end).*x(V_startIdx:end), 1 );
        v = norm( q*x(V_startIdx:end), 1 ) + norm( r*x(1:V_startIdx-1), 1 );
    end

    function x = prox_f(x,t)
        tq = t .* q; % March 2012, allowing vectorized stepsizes
        tr = t .* r; 
        
        % project U
	    %x(1:V_startIdx-1) = max(x(1:V_startIdx-1) - tq, 0);
        s  = 1 - min( tq./abs(x(1:V_startIdx-1)), 1 );
        x(1:V_startIdx-1) = x(1:V_startIdx-1).*s;
        
        % project V
%         x(V_startIdx:end) = max(x(V_startIdx:end) - tr, 0);
        
        s  = 1 - min( tq./abs(x(V_startIdx:end)), 1 );
        x(V_startIdx:end)  = max(x(V_startIdx:end) .* s, 0);
    end


end


