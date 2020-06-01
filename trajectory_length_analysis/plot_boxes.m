%%

h_data = {cell2mat(transpose(arclen_ltc_solver));cell2mat(transpose(arclen_node_solver));cell2mat(transpose(arclen_ctrnn_solver))};
aboxplot(h_data,'labels',{'10','25','50','100','150','200'})

set(gca,'FontSize',15)
%ylim([0 100])
ylabel('Trajectory Length');
xlabel('Network Width (k)');
legend({'LTC','N-ODE','CT-RNN'},'FontSize',14);
legend('boxoff')
set(gca,'TickLength',[0,0])
set(gca,'box','off')
set(gca,'YScale','log');
grid on;
set(gca, 'XGrid', 'off');
ylim([1 1000000])
annotation('textbox',...
    [0.55 0.63 0.7 0.3],...
    'String',{['samples = ' num2str(i)],['solver = RK45'], ['activations = ' activation_], ['depth = ' num2str(n) ',  \sigma^{2}_{w} = ' num2str(weight_dist_variance/k) ',  \sigma^{2}_{b} = ' num2str(1)]},...
    'FontSize',13,...
    'EdgeColor','None');
    %'FontName','Arial',...
    %'LineStyle','--',...
    %'LineWidth',2,...
    %'BackgroundColor',[0.9  0.9 0.9],...
    %'Color',[0.84 0.16 0]

 
%% Plot Sigma variation

h_data = {cell2mat(transpose(arclen_ltc_solver));cell2mat(transpose(arclen_node_solver));cell2mat(transpose(arclen_ctrnn_solver))};
aboxplot(h_data,'labels',{'1','2','4','8','16','32'})

set(gca,'FontSize',15)
%ylim([0 100])
ylabel('Trajectory Length');
xlabel('\sigma_{w}^2');
legend({'LTC','N-ODE','CT-RNN'},'FontSize',14);
legend('boxoff')
set(gca,'TickLength',[0,0])
set(gca,'box','off')
set(gca,'YScale','log');
grid on;
set(gca, 'XGrid', 'off');
ylim([1 10000])
annotation('textbox',...
    [0.55 0.63 0.7 0.3],...
    'String',{['samples = ' num2str(i)],['solver = RK45'], ['activations = ' activation_], ['depth = ' num2str(n) ',  \sigma^{2}_{b} = ' num2str(1)]},...
    'FontSize',13,...
    'EdgeColor','None');
    %'FontName','Arial',...
    %'LineStyle','--',...
    %'LineWidth',2,...
    %'BackgroundColor',[0.9  0.9 0.9],...
    %'Color',[0.84 0.16 0]
    
%% Plot different solvers traj

h_data = {cell2mat(transpose(arclen_ltc_solver));cell2mat(transpose(arclen_node_solver));cell2mat(transpose(arclen_ctrnn_solver))};
aboxplot(h_data,'labels',{'RK2(3)','RK4(5)','ABM1(13)','TR-BDF2'})

set(gca,'FontSize',15)
%ylim([0 100])
ylabel('Trajectory Length');
xlabel('ODE Solvers');
legend({'LTC','N-ODE','CT-RNN'},'FontSize',14);
legend('boxoff')
set(gca,'TickLength',[0,0])
set(gca,'box','off')
% set(gca,'YScale','log');
% grid on;
% set(gca, 'XGrid', 'off');
% ylim([3 100000])
annotation('textbox',...
    [0.55 0.63 0.7 0.3],...
    'String',{['samples = ' num2str(i)], ['activations = ' activation_], ['depth = ' num2str(n) ',  \sigma^{2}_{w} = ' num2str(weight_dist_variance/k) ',  \sigma^{2}_{b} = ' num2str(1)]},...
    'FontSize',12,...
    'EdgeColor','None');
    %'FontName','Arial',...
    %'LineStyle','--',...
    %'LineWidth',2,...
    %'BackgroundColor',[0.9  0.9 0.9],...
    %'Color',[0.84 0.16 0]

%% Impact of Layers

h_data = {arclen_ltc_solver{4,1};arclen_node_solver{4,1};arclen_ctrnn_solver{4,1}};
aboxplot(h_data,'labels',{'L1','L2','L3','L4'})

set(gca,'FontSize',15)
%ylim([0 100])
ylabel('Trajectory Length');
xlabel('Network Layers');
legend({'LTC','N-ODE','CT-RNN'},'FontSize',14);
legend('boxoff')
set(gca,'TickLength',[0,0])
set(gca,'box','off')
set(gca,'YScale','log');
grid on;
set(gca, 'XGrid', 'off');
ylim([3 10000000])
annotation('textbox',...
    [0.55 0.63 0.7 0.3],...
    'String',{['samples = ' num2str(i)],['solver = RK45'], ['activations = ' activation_], ['depth = ' num2str(n) ',  \sigma^{2}_{w} = ' num2str(weight_dist_variance/k) ',  \sigma^{2}_{b} = ' num2str(1)]},...
    'FontSize',13,...
    'EdgeColor','None');
    %'FontName','Arial',...
    %'LineStyle','--',...
    %'LineWidth',2,...
    %'BackgroundColor',[0.9  0.9 0.9],...
    %'Color',[0.84 0.16 0]
    
    
%% Impact of step size

h_data = {cell2mat(transpose(arclen_ltc_solver));cell2mat(transpose(arclen_node_solver));cell2mat(transpose(arclen_ctrnn_solver))};
aboxplot(h_data,'labels',{'0.1','0.2','0.02','0.01','0.001'})

set(gca,'FontSize',15)
%ylim([0 100])
ylabel('Trajectory Length');
xlabel('Input step-size');
legend({'LTC','N-ODE','CT-RNN'},'FontSize',14);
legend('boxoff')
set(gca,'TickLength',[0,0])
set(gca,'box','off')
set(gca,'YScale','log');
grid on;
set(gca, 'XGrid', 'off');
ylim([3 10000000])
annotation('textbox',...
    [0.55 0.63 0.7 0.3],...
    'String',{['samples = ' num2str(i)],['solver = RK45'], ['activations = ' activation_], ['depth = ' num2str(n) ',  \sigma^{2}_{w} = ' num2str(weight_dist_variance/k) ',  \sigma^{2}_{b} = ' num2str(1)]},...
    'FontSize',13,...
    'EdgeColor','None');
    %'FontName','Arial',...
    %'LineStyle','--',...
    %'LineWidth',2,...
    %'BackgroundColor',[0.9  0.9 0.9],...
    %'Color',[0.84 0.16 0]
    
      
% h_data = {cell2mat(transpose(arclen_ltc_solver));cell2mat(transpose(arclen_node_solver));cell2mat(transpose(arclen_ctrnn_solver))};
% aboxplot(h_data,'labels',{'1','2','4','8','150','200'})
% 
% set(gca,'FontSize',15)
% %ylim([0 100])
% ylabel('Trajectory Length');
% xlabel('Network Width (k)');
% legend({'LTC','N-ODE','CT-RNN'},'FontSize',14);
% legend('boxoff')
% set(gca,'TickLength',[0,0])
% set(gca,'box','off')
% set(gca,'YScale','log');
% grid on;
% set(gca, 'XGrid', 'off');
% ylim([3 10000])
% annotation('textbox',...
%     [0.55 0.63 0.7 0.3],...
%     'String',{['samples = ' num2str(i)],['solver = RK4(5)'], ['activations = ' activation_], ['depth = ' num2str(n) ',  \sigma^{2}_{w} = ' num2str(weight_dist_variance/k) ',  \sigma^{2}_{b} = ' num2str(1)]},...
%     'FontSize',12,...
%     'EdgeColor','None');
%     %'FontName','Arial',...
%     %'LineStyle','--',...
%     %'LineWidth',2,...
%     %'BackgroundColor',[0.9  0.9 0.9],...
%     %'Color',[0.84 0.16 0]


% figure
% h_data = {TL_LTC_sigma;TL_ctrnn_sigma;TL_odernn_sigma};
% aboxplot(h_data,'labels',{'1','2','4','8'})
% set(gca,'FontSize',15)
% ylim([0 10e6])
% set(gca,'YScale','log');
% grid on;
% set(gca, 'XGrid', 'off');
% ylabel('Trajectory Length');
% xlabel('Weight Distribution \sigma^{2}');
% legend({'LTC','ODE-RNN','CT-RNN'},'FontSize',14);
% legend('boxoff')
% set(gca,'TickLength',[0,0])
% set(gca,'box','off')
% 
% 
% figure
% h_data = {TL_LTC_E;TL_ctrnn_E;TL_odernn_E};
% aboxplot(h_data,'labels',{'10','25','50','100','150','250'})
% set(gca,'FontSize',15)
% ylim([1 1000])
% ylabel('Trajectory Length');
% xlabel('Network Width (k)');
% legend({'LTC','ODE-RNN','CT-RNN'},'FontSize',14);
% legend('boxoff')
% set(gca,'TickLength',[0,0])
% set(gca,'box','off')
% 
% 
% figure
% h_data = {TL_LTC_E_pn;TL_ctrnn_E_pn;TL_odernn_E_pn};
% aboxplot(h_data,'labels',{'2','10','25','50','100','150','250'})
% set(gca,'FontSize',15)
% ylim([1 1000000])
% set(gca,'YScale','log');
% grid on;
% set(gca, 'XGrid', 'off');
% ylabel('Trajectory Length');
% xlabel('Network Width (k)');
% legend({'LTC','ODE-RNN','CT-RNN'},'FontSize',14);
% legend('boxoff')
% set(gca,'TickLength',[0,0])
% set(gca,'box','off')
% 
% figure
% h_data = {TL_LTC_sigma_E_pn;TL_ctrnn_sigma_E_pn;TL_odernn_sigma_E_pn};
% aboxplot(h_data,'labels',{'1','2','4','8'})
% set(gca,'FontSize',15)
% ylim([1 10e6])
% set(gca,'YScale','log');
% grid on;
% set(gca, 'XGrid', 'off');
% ylabel('Trajectory Length');
% xlabel('Weight Distribution \sigma^{2}');
% legend({'LTC','ODE-RNN','CT-RNN'},'FontSize',14);
% legend('boxoff')
% set(gca,'TickLength',[0,0])
% set(gca,'box','off')
% 
% 
% figure
% h_data = {TL_LTC_step;TL_ctrnn_step;TL_odernn_step};
% aboxplot(h_data,'labels',{'1','10','25','50','100','1000'})
% set(gca,'FontSize',15)
% ylim([1 10e6])
% set(gca,'YScale','log');
% grid on;
% set(gca, 'XGrid', 'off');
% ylabel('Trajectory Length');
% xlabel('ODE Solver Step Size');
% legend({'LTC','ODE-RNN','CT-RNN'},'FontSize',14);
% legend('boxoff')
% set(gca,'TickLength',[0,0])
% set(gca,'box','off')
