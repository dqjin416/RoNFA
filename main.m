clc
clear

sampletype='balance';
nways=5;
kshot=5;
sample_num=20;
times=1;
new_acc=zeros(times+1,1);
org_acc=zeros(times+1,1);
accuracy=zeros(times+1,1);
all_time_new=zeros(times,1);
all_time_org=zeros(times,1);
test_time_new=zeros(times,1);
test_time_org=zeros(times,1);
total_time=zeros(times,1);
deleted_noisy_counts = zeros(times+1, 1); % 存储每次运行的删除噪声样本数量
corrected_noisy_counts = zeros(times+1, 1); % 存储每次运行的修正噪声样本数量

start_time = tic;
% 定义噪音比例
noise_rate = 0.4; % 20%的噪音
noise_type = 'outlier'; % pair\sym\outlier
d_name = "mini_test";
net = "conv4";
% 加载离群数据
load('./data/outlier_val_' + net + '_features.mat');
load('./data/outlier_val_labels.mat');
out_feature = features;
out_label = labels;
load('./data/' + d_name + '_' + net + '_features.mat');
load('./data/' + d_name + '_labels.mat');

for z= 1:times
    dataset_loader
    % 定义mask_indices，假设支持集的索引为1:n_support
    mask_indices.support = 1:25; % 支持集的样本索引
    [L,ratioL]=Sample(instance,instance_label,kshot,sampletype);
    [mt,~]=size(instance);
    UL=setdiff((1:mt)',L);%未被采样的样本索引
    S_set=instance(L,:);
    S_set_label=instance_label(L,:);
    Q_set=instance(UL,:);
    Q_set_label=instance_label(UL,:);
    XT_KNN=Q_set;
    YT=Q_set_label;
    [Y_data,Y_noisy, noise_positions] = add_noise(S_set,S_set_label, mask_indices, nways, noise_rate, noise_type,out_feature,out_label);
    [VSS,VSS_Y]=VSSCon_all8(S_set,Y_noisy);
    
    tic
    [Q_set_label_pre1,test_time_new(z)]=MSNMclassifier(VSS,VSS_Y,Q_set);
    all_time_new(z)=toc;
    new_acc(z)=mean(Q_set_label_pre1==Q_set_label);
    fprintf('This is the %d th running \n',z);
    disp(['The accuracy by clustering =',num2str(new_acc(z))])

end
% 计算整个过程的总时间
total_time = toc(start_time);  % 计算从开始到第二次MSNMclassifier的时间

disp(['The ', num2str(nways),'way-',num2str(kshot),'shot newacc of ',d_name,' = ',num2str(mean(new_acc))])
disp(['The testing new time by clustering=',num2str(mean(test_time_new))])
disp(['The origin train time =',num2str(mean(all_time_org-test_time_org))])
disp(['The time =',num2str(mean(total_time))])

new_acc(times+1) = mean(new_acc(1:times));