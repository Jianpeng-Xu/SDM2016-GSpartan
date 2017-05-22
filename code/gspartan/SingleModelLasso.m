function w = SingleModelLasso(X, Y, lambda)
% wrapper for leastR
% w = ridge(Y, X, lambda);
addpath(genpath('../SLEP/SLEP')); % load function
rho=lambda;            % the regularization parameter
                    % it is a ratio between (0,1), if .rFlag=1

%----------------------- Set optional items ------------------------
opts=[];

% Starting point
opts.init=2;        % starting from a zero point

% termination criterion
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=100;   % maximum number of iterations

% normalization
opts.nFlag=0;       % without normalization

% regularization
opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)
%opts.rsL2=0.01;     % the squared two norm term

%----------------------- Run the code LeastR -----------------------
% fprintf('\n mFlag=0, lFlag=0 \n');
% opts.mFlag=0;       % treating it as compositive function 
% opts.lFlag=0;       % Nemirovski's line search
% tic;
% [w, funVal1, ValueL1]= LeastR(X, Y, rho, opts);
% toc;

% opts.maxIter=1000;
% 
% fprintf('\n mFlag=1, lFlag=0 \n');
% opts.mFlag=1;       % smooth reformulation 
% opts.lFlag=0;       % Nemirovski's line search
% opts.tFlag=2; opts.tol= funVal1(end);
% tic;
% [x2, funVal2, ValueL2]= LeastR(A, y, rho, opts);
% toc;
% 
% fprintf('\n mFlag=1, lFlag=1 \n');
opts.mFlag=1;       % smooth reformulation 
opts.lFlag=1;       % adaptive line search
% opts.tFlag=2; opts.tol= funVal1(end);
% tic;
% [x3, funVal3, ValueL3]= LeastR(A, y, rho, opts);
[w, funVal1, ValueL1]= LeastR(X, Y, rho, opts);