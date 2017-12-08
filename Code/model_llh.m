function [LLH] = model_llh(params, data, N, T)
%params=parameters(1,:);
%params=prop_param;

p.rho1 = params(1);
p.rho2=params(2);
p.phi1 = params(3);
p.phi2=params(4);
p.beta=params(5);
p.sigma_eps = params(6);
p.sigma_1 = params(7);
p.sigma_2 = params(8);

T = min(T, length(data));

%%% Model-implied transition equations:
%%%long-run dist
%X1=0;
%X2=0;
%eps1=normrnd(0,params(5));
%eps2=normrnd(0,params(5));
%state=zeros(1,6);
%for i = 1:10000;
%eps=normrnd(0,params(6));
%state(1,1)=[X1,X2,eps1,eps2]*params(1:4)'+eps;
%state(1,2)=X1;
%state(1,5)=X2;
%state(1,3)=eps;
%state(1,4)=eps1;
%state(1,6)=eps2;
%X1=state(1);
%X2=state(2);
%eps1=state(3);
%eps2=state(4);
%end
particles=zeros(T,N,6);
state=[0,0,normrnd(0,params(6)),normrnd(0,params(6))];
epsi1=normrnd(0,params(6),[10000,1]);
epsi2=normrnd(0,params(6),[10000,1]);
long_run_dist(:,2)=state*params(1:4)'+epsi2;
long_run_dist(:,1)=long_run_dist(:,1)*params(1)+state(1)*params(2)+epsi1*params(3)+state(3)*params(4)+epsi1;
long_run_dist(:,3)=epsi1;
long_run_dist(:,4)=epsi2;
long_run_dist(:,5)=state(1);
long_run_dist(:,6)=normrnd(0,params(6));
%%% Empirical log-likelihoods by particle filtering
% initialize particles according to S_0

rng(0)
init_seeds = randi([1,10000],[N,1]);

particles = zeros(T, N, 6);
llhs = zeros(T,1);

particles(1, : ,1:6) = permute(long_run_dist(init_seeds,:),[3 1 2]);

llhs(1) = log( mean( exp( ...
        log( normpdf(log(data(1,:,1)), particles(1,:,1), p.sigma_1) ) + ...
        log( normpdf(data(1,:,2), p.beta*particles(1,:,1).^2, p.sigma_2) ) ...
        ) ) );

% predict, filter, update particles and collect the likelihood 

for t = 2:T
    %%% Prediction:
    shock = normrnd(0,p.sigma_eps,[1 N+1]);
    for n = 1:N
        particles(t,n,1) = permute(particles(t-1,n,1:4),[1 3 2])*params(1:4)'+shock(1,n) ;
        particles(t,n,2)=particles(t-1,n,1);
        particles(t,n,3)=shock(1,n);
        particles(t,n,4)=particles(t-1,n,3);
        particles(t,n,5)=particles(t-1,n,2);
        particles(t,n,6)=shock(1,n+1);
    end
    
    %%% Filtering:
    llh = log( normpdf(log(data(t,:,1)), particles(t,:,1),  p.sigma_1) ) + ...
        log( normpdf(data(t,:,2), p.beta*particles(t,:,1).^2, p.sigma_2) );
    llh1=llh;
    llh1(llh1==-inf)=0;
    lh1 = exp(llh1);
    lh=exp(llh);
    weights = exp( llh1 - log( sum(lh1) ) );
    % store the log(mean likelihood)
    llhs(t) = log(mean(lh));
    
    %%% Sampling:
    particles(t,:,1:6) = datasample(particles(t,:,1:6), N, 'Weights', weights);
    
end

LLH = sum(llhs);