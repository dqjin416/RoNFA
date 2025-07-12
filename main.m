clc
clear

sampletype='balance';
dataset='FC100';
nways=5;
kshot=5;
sample_num=20;
times=600;
new_acc=zeros(times,1);
org_acc=zeros(times,1);
rnnp_acc=zeros(times,1);
accuracy=zeros(times,1);
svm_acc=zeros(times,1);
knn_acc=zeros(times,1);
newPCA_acc=zeros(times,1);
orgPCA_acc=zeros(times,1);
pcanet_knn_acc=zeros(times,1);
pcanet_svm_acc=zeros(times,1);
all_time_new=zeros(times,1);
all_time_org=zeros(times,1);
test_time_new=zeros(times,1);
test_time_org=zeros(times,1);
deleted_noisy_counts = zeros(times, 1); % 存储每次运行的删除噪声样本数量
corrected_noisy_counts = zeros(times, 1); % 存储每次运行的修正噪声样本数量

% %%%%%%%%%%%%%%%%%Parameters of PCANET
% PCANet.NumStages = 2;
% PCANet.PatchSize = [3 3];
% PCANet.NumFilters = [4 7] ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
% PCANet.HistBlockSize = [3 7];
% PCANet.BlkOverLapRatio = 0;
% PCANet.Pyramid = [ ];
% ImgFormat = 'gray';
% PCANet;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%

% 定义噪音比例
noise_rate = 0.6; % 20%的噪音
noise_type = 'outlier';
for z= 1:times
    if strcmp(dataset,'cub')
        cub;
    elseif strcmp(dataset,'cifar100')
        cifar100;
    elseif strcmp(dataset,'miniimage')
        miniimage;
    elseif strcmp(dataset,'tiredimage')
        tiredimage;
    elseif strcmp(dataset,'FC100')
        FC100;
    end
    
    % 定义mask_indices，假设支持集的索引为1:n_support
    mask_indices.support = 1:25; % 支持集的样本索引
    % mask_indices.noise = []; % 如果特征数据无噪声，则无需定义噪声集
    [L,ratioL]=Sample(instance,instance_label,kshot,sampletype);
    % [L,ratioL,Y_support, Y_noisy, Q_set_label]=Sample_with_noise(instance,instance_label,kshot,sampletype,noise_rate);
    % [COEFF, SCORE, LATENT, TSQUARED] = pca((instance)) ;
    % instance=SCORE(:,1:4);
    % instance = tsne(instance, 'NumDimensions', 10);
    % ica = rica(instance, 10); % 使用 10 个独立成分
    % instance = transform(ica, instance);
    [mt,~]=size(instance);
    UL=setdiff((1:mt)',L);%未被采样的样本索引
    S_set=instance(L,:);
    S_set_label=instance_label(L,:);
    Q_set=instance(UL,:);
    Q_set_label=instance_label(UL,:);
    % KNN_XL=S_set;
    % YL=S_set_label;
    XT_KNN=Q_set;
    YT=Q_set_label;
    [Y_data,Y_noisy, noise_positions] = add_noise(S_set,S_set_label, mask_indices, nways, noise_rate, noise_type);
    % 添加对称噪音到标签
    % Y_noisy = addSymmetricNoise(S_set_label, noise_rate);
    % % 添加非对称噪音到标签
    % Y_noisy = addAsymmetricNoise(S_set_label, nways, kshot, noise_rate)；
    
    % S_set_label = Y_noisy;

    
    % [~,~,~,instance,S_set_ORG]=IniSig(S_set,instance,L);
    % sigm1=1;

    % [Labeled_X, Labeled_Y, Unlabeled_X, Unlabeled_Y]=select(S_set, S_set_label, 0.1);
    [VSS,VSS_Y]=VSSCon_all8(S_set,Y_noisy);
    % [VSS,VSS_Y,deleted_noisy_count, corrected_noisy_count]=VSSCon_all7(S_set,Y_noisy,S_set_label);
    % deleted_noisy_counts(z) = deleted_noisy_count; % 记录删除的噪声数量
    % corrected_noisy_counts(z) = corrected_noisy_count; % 记录修正的噪声数量
    % [VSS,VSS_Y]=VSSCon_all6(S_set,S_set_label);
    % [VSS1,VSS_Y1]=VSSCon(S_set,S_set_label);
    % % 绘制聚类结果的散点图
    % figure;
    % hold on;
    % title('聚类后的散点图');
    % xlabel('特征维度 1');
    % ylabel('特征维度 2');
    % 
    % % 获取唯一标签并生成不同颜色
    % unique_labels = unique(VSS_Y);
    % num_labels = length(unique_labels);
    % colors = lines(num_labels); % 使用lines生成调色板
    % 
    % % 为每个标签类别绘制聚类点
    % for i = 1:num_labels
    %     label = unique_labels(i);
    %     label_points = ins(VSS_Y == label, :); % 获取当前标签的特征点
    %     scatter(label_points(:, 1), label_points(:, 2), 50, 'filled', ...
    %             'MarkerFaceColor', colors(i, :), 'DisplayName', sprintf('标签 %d', label));
    % end
    % 
    % legend('show'); % 显示图例
    % % 保存为 JPG 图片
    % saveas(gcf, 'clustering_results.jpg');
    % hold off;
    % [VSS,VSS_Y, XL_new, YL_new]=VSSConnoisy1(S_set,Y_noisy);
    
    tic

    [Q_set_label_pre,test_time_org(z)]=MSNMclassifier(S_set,S_set_label,Q_set);
    [Q_set_label_pre1,test_time_new(z)]=MSNMclassifier(VSS,VSS_Y,Q_set);
    % [Q_set_label_pre1,test_time_new(z)]=MSNMclassifier_initial(VSS,VSS_Y,Q_set);
    predictions = RNNP1(S_set, Y_noisy, Q_set);
    KNN_XL=VSS;
    YL=VSS_Y;

    all_time_org(z)=toc;    
    org_acc(z)=mean(Q_set_label_pre==Q_set_label);
    fprintf('This is the %d th running \n',z);
    disp(['The accuracy =',num2str(org_acc(z))])

    all_time_new(z)=toc;
    new_acc(z)=mean(Q_set_label_pre1==Q_set_label);
    fprintf('This is the %d th running \n',z);
    disp(['The accuracy by clustering =',num2str(new_acc(z))])

    rnnp_acc(z)=mean(predictions==Q_set_label);
    fprintf('This is the %d th running \n',z);
    disp(['The accuracy by rnnp =',num2str(rnnp_acc(z))])

    [~,~,accuracy(z)] = protonet(S_set, Y_noisy, Q_set, Q_set_label,nways,kshot);
    fprintf('This is the %d th running \n',z);
    disp(['The accuracy by protonet =',num2str(accuracy(z))])
    % % knnduibi;
    % svmduibi;

    
end
disp(['The ', num2str(nways),'way-',num2str(kshot),'shot orgacc of ',dataset,' = ',num2str(mean(org_acc))])
disp(['The ', num2str(nways),'way-',num2str(kshot),'shot newacc of ',dataset,' = ',num2str(mean(new_acc))])
disp(['The ', num2str(nways),'way-',num2str(kshot),'shot newacc of ',dataset,' = ',num2str(mean(rnnp_acc))])
disp(['The ', num2str(nways),'way-',num2str(kshot),'shot newacc of ',dataset,' = ',num2str(mean(accuracy))])
disp(['The knn accuracy =',num2str(mean(knn_acc))])
disp(['The svm accuracy =',num2str(mean(svm_acc))])
disp(['The testing new time by clustering=',num2str(mean(test_time_new))])
disp(['The origin train time =',num2str(mean(all_time_org-test_time_org))])
% 创建表格保存准确性结果
results_table = table((1:times)', org_acc, new_acc, rnnp_acc,accuracy, deleted_noisy_counts, corrected_noisy_counts,...
    'VariableNames', {'Run', 'Original_Accuracy', 'New_Accuracy', 'rnnp_acc','protonet_acc','deleted_noisy_count', 'corrected_noisy_count'});

% 将表格保存为 CSV 文件
writetable(results_table, 'accuracy_results.csv');

% 显示表格
disp(results_table);