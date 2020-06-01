% run this file for generating Trajectory length of continuous-time models.

for ii=1:1 % this is to loop over different modes
    % 1:4 different layers
    % 1:4 different sigma
    % 1:4 different types of activations
    % 1:6 number of neurons in one layer 
    % set ii = 1:4 for solver selection, uncomment the switch condition.

for i = 1:1 % number of samples

%number_of_input_steps = [10 20 50 100 1000];
steps =100;

t_max = 2*pi;

% Input delta-t
delta_t = t_max /steps;
tspan = [0 t_max];

% Define a circular trajectory
gt = linspace(0,t_max,steps);
inputs = [sin(gt);cos(gt)];
m = size(inputs,1);
%Seed control
%rng(1);

%num_of_layers = [1 2 3 4];
n = 1; %number of layers

%num_of_neurons = [10 25 50 100 150 200];
k = 25; % number of neurons

total_neurons = n*k;

%initialized ODEs by zeros
ic = zeros(1,total_neurons);

%Choice of activations:
%activations = {'tanh', 'sigmoid', 'relu', 'Htanh'};
%activation_ = char(activations(ii));
activation_ = 'Htanh';

%sigma_papa = [1 2 4 8 16 32];
sigma = 2; 
sigma_b = 1;
weight_dist_variance = k*sigma;
p_weights = makedist('normal','mu',0,'sigma',weight_dist_variance/k);
p_biases = makedist('normal','mu',0,'sigma',sigma_b);

% p_weights = makedist('poisson','lambda',weight_dist_variance/k);
% p_biases = makedist('poisson','lambda',10);

% forward weight initializaition
w_l = 0 +1 * random(p_weights,[n,k,k]);
b_l = 0 +1 * random(p_biases,[n,k,1]);

% E_l stands for A in Eq. 1 of the paper.
E_l = 0 * ones(n,k,k) + 1 * random(p_weights,[n,k,k]);
tau = 0*ones(total_neurons,1) +1*abs(random(p_biases,[total_neurons,1]));

%recurrent weights initialization
w_l_r = 0 +1 * random(p_weights,[n,k,k]);
b_l_r = 0 +1 * random(p_biases,[n,k,1]);
E_l_r = 0 * ones(n,k,k) + 1* random(p_weights,[n,k,k]);

% Set solvers tolorance
opts = odeset('RelTol',1e-2,'AbsTol',1e-4);

% Uncomment if you wanna try different advanced solvers:
% switch ii
%     case ii==1
%         solver = 'ode23';
%         [t_node,y_node] = ode23(@(t,y) node_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,total_neurons,k,activation_), tspan, ic, opts);
%         [t_ctrnn,y_ctrnn] = ode23(@(t,y) ctrnn_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);
%         [t_ltc,y_ltc] = ode23(@(t,y) ltc_def(t,y,gt,inputs,w_l,b_l,E_l,w_l_r,b_l_r,E_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);
%     case ii==2
%         solver = 'ode45';
%         [t_node,y_node] = ode45(@(t,y) node_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,total_neurons,k,activation_), tspan, ic, opts);
%         [t_ctrnn,y_ctrnn] = ode45(@(t,y) ctrnn_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);
%         [t_ltc,y_ltc] = ode45(@(t,y) ltc_def(t,y,gt,inputs,w_l,b_l,E_l,w_l_r,b_l_r,E_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);
%     case ii==2
%         solver = 'ode113';
%         [t_node,y_node] = ode113(@(t,y) node_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,total_neurons,k,activation_), tspan, ic, opts);
%         [t_ctrnn,y_ctrnn] = ode113(@(t,y) ctrnn_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);
%         [t_ltc,y_ltc] = ode113(@(t,y) ltc_def(t,y,gt,inputs,w_l,b_l,E_l,w_l_r,b_l_r,E_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);
%     case ii==3
%         solver = 'ode23tb';
%         [t_node,y_node] = ode23tb(@(t,y) node_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,total_neurons,k,activation_), tspan, ic, opts);
%         [t_ctrnn,y_ctrnn] = ode23tb(@(t,y) ctrnn_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);
%         [t_ltc,y_ltc] = ode23tb(@(t,y) ltc_def(t,y,gt,inputs,w_l,b_l,E_l,w_l_r,b_l_r,E_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);
% 
% end


% defult solver:
solver = 'ode45';
[t_node,y_node] = ode45(@(t,y) node_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,total_neurons,k,activation_), tspan, ic, opts);
[t_ctrnn,y_ctrnn] = ode45(@(t,y) ctrnn_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);
[t_ltc,y_ltc] = ode45(@(t,y) ltc_def(t,y,gt,inputs,w_l,b_l,E_l,w_l_r,b_l_r,E_l_r,tau,total_neurons,k,activation_), tspan, ic, opts);

% uncomment if you wanna run a fixed step-size solver
% solver = 'ode1';
% tspan = linspace(0,t_max,steps);
% y_node = ode1(@(t,y) node_def_fixedstep(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,total_neurons,k,activation_), tspan, ic);
% y_ctrnn = ode1(@(t,y) ctrnn_def_fixedstep(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,tau,total_neurons,k,activation_), tspan, ic);
% y_ltc = ode1(@(t,y) ltc_def_fixedstep(t,y,gt,inputs,w_l,b_l,E_l,w_l_r,b_l_r,E_l_r,tau,total_neurons,k,activation_), tspan, ic);

%% Principle component analysis + computing of the trajectory length

for j= 1:n 

[a1,b1,c1,su1,explained1] = pca(y_node(:,(j-1)*k+1:j*k));
[a2,b2,c2,su2,explained2] = pca(y_ctrnn(:,(j-1)*k+1:j*k));
[a3,b3,c3,su3,explained3] = pca(y_ltc(:,(j-1)*k+1:j*k));

[arclen_input_solver{ii,1}(i,j),] = arclength(inputs(1,:),inputs(2,:));
[arclen_node_solver{ii,1}(i,j),] = arclength(b1(:,1),b1(:,2));
[arclen_ctrnn_solver{ii,1}(i,j),] = arclength(b2(:,1),b2(:,2));
[arclen_ltc_solver{ii,1}(i,j),] = arclength(b3(:,1),b3(:,2));

[explained_node_solver{ii,i}(:,j),] = explained1;
[explained_ctrnn_solver{ii,i}(:,j),] = explained2;
[explained_ltc_solver{ii,i}(:,j),] = explained3;

subplot(1,n,j)
plot(inputs(1,:),inputs(2,:),'r','LineWidth',1)
hold on;
plot(b1(:,1),b1(:,2),'b','LineWidth',1)
plot(b2(:,1),b2(:,2),'g','LineWidth',1)
plot(b3(:,1),b3(:,2),'k','LineWidth',1)
hold off;
title({['l(N-ODE) = ' num2str(arclen_node_solver{ii,1}(i,j))],['l(CT-RNN) = ' num2str(arclen_ctrnn_solver{ii,1}(i,j))],['l(LTC) = ' num2str(arclen_ltc_solver{ii,1}(i,j))]});
set(0,'DefaultAxesTitleFontWeight','normal');
set(gca,'FontSize',12)
%axis('off')
%ylim([100 100000])
%xticks([0 1]);
xticklabels({});
yticklabels({});
ylabel('2^{nd} Latent Dimension');
xlabel('1^{st} Latent Dimension');
legend({'Inputs','N-ODE','CT-RNN','LTC'},'FontSize',10,'Location','southeastoutside');
legend('boxoff')

if ~isfile(['figs/' activation_ '_' num2str(k)])
    folder = mkdir(['figs/']);
end

saveas(gcf,['figs/'  solver '_' activation_ '_sigma_' num2str(weight_dist_variance/k) '_n_' num2str(n) '_' 'k_' num2str(k) '_layer_' num2str(j) '_sample_' num2str(i) '.eps'])
end

% record sampled time-stamps taken by the solver
sampling_time_ltc{ii,i} = t_ltc;
sampling_time_node{ii,i} = t_node;
sampling_time_ctrnn{ii,i} = t_ctrnn;

%% Compute the average number of integration steps of solvers for all models
% This section first computes how many integration steps has been computed 
% between every two consequetive arriving input samples, and then computes
% the average of those for the entire sequence length.

steps_ltc = zeros(size(gt,2),1);
steps_node = zeros(size(gt,2),1);
steps_ctrnn = zeros(size(gt,2),1);

for ll = 1:size(gt,2)-1
    for jj = 1:size(t_ltc,1)
    if  gt(1,ll) < t_ltc(jj,1) && t_ltc(jj,1) <= gt(1,ll+1)
        steps_ltc(ll,1) = steps_ltc(ll,1)+1;
    end      
    end
    
    for jj = 1:size(t_ctrnn,1)
    if gt(1,ll) < t_ctrnn(jj,1) && t_ctrnn(jj,1) <= gt(1,ll+1)
        steps_ctrnn(ll,1) = steps_ctrnn(ll,1)+1;
    end      
    end
    
    for jj = 1:size(t_node,1)
    if gt(1,ll) < t_node(jj,1) && t_node(jj,1) <= gt(1,ll+1)
        steps_node(ll,1) = steps_node(ll,1)+1;
    end      
    end
end

avg_steps{ii,i} = [mean(steps_node) mean(steps_ctrnn) mean(steps_ltc)];
std_steps{ii,i} = [std(steps_node) std(steps_ctrnn) std(steps_ltc)];

%% Compute Theoretical lower bounds - Presented in Section 5 of "Liquid time-constant networks" paper:
% IMPORTANT NOTE: Lower bounds are computed only for the RNN models were
% they DO NOT recurrently synapse into each other. Therefore before running
% this section, you must make sure you have set all recurrent weights
% to zero.
% lower_bound_node =  (((sqrt(k*sigma))/((sqrt(sigma+sigma_b + (k * sqrt(sigma+sigma_b))))))^((1)*n*mean(steps_node)))* 6.2821;
% lower_bound_ctrnn = (((sqrt(sqrt(1)*k*sigma))/((sqrt(sigma+sigma_b + (k * sqrt(sigma+sigma_b))))))^((1)*n*mean(steps_ctrnn)))* 6.2821;
% lower_bound_ltc = (((sqrt(k*(sigma-sigma_b)))/((sqrt(sigma+sigma_b + k * sqrt(sigma+sigma_b)))))^(n) * abs((sigma + min((1/mean(steps_ltc)),1/steps)*norm((b3(:,1)+b3(:,2))/(2)))))* 6.2821;
% 
% disp(['Neural ODE Length: ' num2str(arclen_node_solver{ii,1}(i,j)) ' CT-RNN Length: ' num2str(arclen_ctrnn_solver{ii,1}(i,j)) ' LTC Length: ' num2str(arclen_ltc_solver{ii,1}(i,j))])
% disp(['Lower bound Neural ODE: ' num2str(lower_bound_node) ' Lower bound CT-RNN: ' num2str(lower_bound_ctrnn) ' Lower bound LTC: ' num2str(lower_bound_ltc)])
end
end
