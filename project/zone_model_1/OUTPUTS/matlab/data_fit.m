raw_data200 = csvread('200.csv',1,0);
R_t200 = raw_data200(:,1);
A_f200 = raw_data200(:,2);
fuel200 = raw_data200(:,4);

raw_data175 = csvread('175.csv',1,0);
R_t175 = raw_data175(:,1);
A_f175 = raw_data175(:,2);
fuel175 = raw_data175(:,4);

raw_data150 = csvread('150.csv',1,0);
R_t150 = raw_data150(:,1);
A_f150 = raw_data150(:,2);
fuel150 = raw_data150(:,4);

raw_data125 = csvread('125.csv',1,0);
R_t125 = raw_data125(:,1);
A_f125 = raw_data125(:,2);
fuel125 = raw_data125(:,4);

raw_data100 = csvread('100.csv',1,0);
R_t100 = raw_data100(:,1);
A_f100 = raw_data100(:,2);
fuel100 = raw_data100(:,4);

raw_data075 = csvread('075.csv',1,0);
R_t075 = raw_data075(:,1);
A_f075 = raw_data075(:,2);
fuel075 = raw_data075(:,4);

raw_data050 = csvread('050.csv',1,0);
R_t050 = raw_data050(:,1);
A_f050 = raw_data050(:,2);
fuel050 = raw_data050(:,4);

raw_data025 = csvread('025.csv',1,0);
R_t025 = raw_data025(:,1);
A_f025 = raw_data025(:,2);
fuel025 = raw_data025(:,4);




figure1 = figure('InvertHardcopy','off','PaperUnits','points',...
    'Color',[1 1 1]);

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create scatter3
scatter3(R_t025,A_f025,fuel025,S1,C1,'DisplayName','15 minutes',...
    'MarkerFaceColor',[0 0.800000011920929 0.800000011920929],...
    'MarkerEdgeColor',[0 0 0]);

% Create scatter3
scatter3(R_t1,A_f1,fuel2,S1,C2,'DisplayName','30 minutes',...
    'MarkerFaceColor',[0 0.800000011920929 0.200000002980232],...
    'MarkerEdgeColor',[0 0 0]);

% Create scatter3
scatter3(R_t1,A_f1,fuel3,S1,C3,'DisplayName','60 minutes',...
    'MarkerFaceColor',[1 0.600000023841858 0],...
    'MarkerEdgeColor',[0 0 0]);

% Create scatter3
scatter3(R_t1,A_f1,fuel4,S1,C4,'DisplayName','45 minutes',...
    'MarkerFaceColor',[0.800000011920929 0.800000011920929 0],...
    'MarkerEdgeColor',[0 0 0]);

% Create scatter3
scatter3(R_t1,A_f1,fuel5,S1,C5,'DisplayName','75 minutes',...
    'MarkerFaceColor',[1 0.400000005960464 0.400000005960464],...
    'MarkerEdgeColor',[0 0 0]);

% Create scatter3
scatter3(R_t1,A_f1,fuel6,S1,C6,'DisplayName','90 minutes',...
    'MarkerFaceColor',[1 0.200000002980232 0.400000005960464],...
    'MarkerEdgeColor',[0 0 0]);

% Create scatter3
scatter3(R_t1,A_f1,fuel7,S1,C7,'DisplayName','105 minutes',...
    'MarkerFaceColor',[1 0 1],...
    'MarkerEdgeColor',[0 0 0]);

% Create scatter3
scatter3(R_t1,A_f1,fuel8,S1,C8,'DisplayName','120 minutes',...
    'MarkerFaceColor',[0.800000011920929 0 0.800000011920929],...
    'MarkerEdgeColor',[0 0 0]);

% Create xlabel
xlabel('Timber area ratio [%]','FontSize',8);

% Create zlabel
zlabel('Fuel consumption [J]','FontSize',8);

% Create ylabel
ylabel('Furnace size [m sq.]','FontSize',8);

% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes1,[0 0.3]);
% Uncomment the following line to preserve the Y-limits of the axes
% ylim(axes1,[60 160]);
% Uncomment the following line to preserve the Z-limits of the axes
% zlim(axes1,[0 300000000000]);
view(axes1,[-49.0999999999999 16.4]);
box(axes1,'on');
grid(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontName','Palatino','FontSize',8);
% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.686520376175549 0.427932960893855 0.238769464277655 0.217138451100462],...
    'FontSize',8);