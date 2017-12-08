load data
data=permute(data, [1 3 2]);



N = 100; % number of particles
T = 400; % length of time series (given by data)

% data needs to be provided:
% data = 

% priors:
prior.rho1 = @(x) unifpdf(x,0,1);
prior.rho2 = @(x) unifpdf(x,0,1);
prior.phi1=@(x) unifpdf(x,0,1);
prior.phi2=@(x) unifpdf(x,0,1);
prior.beta=@(x) unifpdf(x,4,7);
prior.sigma_eps = @(x) lognpdf(x, -1/2, 1);
prior.sigma_1 = @(x) lognpdf(x, -1/2, 1);
prior.sigma_2 = @(x) lognpdf(x, -1/2, 1);
prior.all = @(p) log(prior.rho1(p(1)))+log(prior.rho1(p(1)))+log(prior.phi1(p(1))) +log(prior.phi2(p(1)))+ ...
    log(prior.sigma_eps(p(2))) + log(prior.sigma_1(p(3))) + log(prior.sigma_2(p(4)));

% proposals according to random walk with parameter sd's:
prop_sig.rho1 = 0.05;
prop_sig.rho2 = 0.05;
prop_sig.phi1 = 0.05;
prop_sig.phi2 = 0.05;
prop_sig.beta=0.05;
prop_sig.sigma_eps = 0.05;
prop_sig.sigma_1 = 0.05;
prop_sig.sigma_2 = 0.05;

prop_sig.all = [prop_sig.rho1 prop_sig.rho2 prop_sig.phi1 prop_sig.phi2 prop_sig.beta prop_sig.sigma_eps prop_sig.sigma_1 prop_sig.sigma_2];

% initial values for parameters
init_params = [0.1 0.1 0.2 0.3 4.5 1.2 0.8 1.8];

% length of sample
M = 5000;
acc_rate = zeros(M,1);

llhs = zeros(M,1);
parameters = zeros(M,8);
parameters(1,:) = init_params;

% evaluate model with initial parameters
log_prior = prior.all(parameters(1,:));
llh = model_llh(parameters(1,:), data, N, T);
llhs(1) = log_prior + llh;

% sample:
rng(0)
proposal_chance = log(rand(M,1));
prop_step = randn(M,8);
for m = 2:M
    % proposal draw:
    prop_param = parameters(m-1,:) + prop_step(m,:) .* prop_sig.all;
    
    % evaluate prior and model with proposal parameters:
    prop_prior = prior.all(prop_param);
    if prop_prior > -Inf % theoretically admissible proposal
        prop_llh = model_llh(prop_param, data, N, T);
        llhs(m) = prop_prior + prop_llh;
        if llhs(m) - llhs(m-1) > proposal_chance(m)
            accept = 1;
        else
            accept = 0;
        end
    else % reject proposal since disallowed by prior
        accept = 0;
    end
    
    % update parameters (or not)
    if accept
        parameters(m,:) = prop_param;
        acc_rate(m) = 1;
    else
        parameters(m,:) = parameters(m-1,:);
        llhs(m) = llhs(m-1);
    end
    sum(acc_rate)/m
    waitbar(m/M)
end
subplot(2,4,1)
hist(parameters(:,1),50)
title('rho1')
subplot(2,4,2)
hist(parameters(:,2),50)
title('rho2')
subplot(2,4,3)
hist(parameters(:,3),50)
title('phi1')
subplot(2,4,4)
hist(parameters(:,4),50)
title('phi2')
subplot(2,4,5)
hist(parameters(:,5),50)
title('beta')
subplot(2,4,6)
hist(parameters(:,6),50)
title('sigma eps')
subplot(2,4,7)
hist(parameters(:,7),50)
title('sigma A')
subplot(2,4,8)
hist(parameters(:,8),50)
title('sigma B')
toc