function predictions = RNNP1(S_set, S_set_label, Q_set)
    % 参数初始化
    alpha = 0.8; % 混合特征比例
    num_mixed_features = 4; % 每类生成的混合特征数量
    unique_labels = unique(S_set_label); % 获取支持集中的唯一类别
    num_classes = numel(unique_labels); % 类别数
    mixed_features = [];
    mixed_labels = [];
    
    % 为每一类生成混合特征
    for c = 1:num_classes
        % 获取当前类别的样本
        class_indices = find(S_set_label == unique_labels(c));
        class_features = S_set(class_indices, :);
        num_samples = size(class_features, 1);
        pairs = nchoosek(1:num_samples, 2); % 生成所有两两组合
        pairs = pairs(randperm(size(pairs, 1)), :); % 随机打乱顺序
        % 生成不重复的混合特征
        % generated_pairs = []; % 用于存储已生成的样本对
        for k = 1:min(num_mixed_features, size(pairs, 1))
            % 随机选择两个不同的样本
            idx_i = randi(num_samples);
            idx_j = randi(num_samples);
            while idx_j == idx_i
                idx_j = randi(num_samples);
            end
            % generated_pairs = [generated_pairs; idx_i, idx_j]; % 记录已生成的样本对
            
            % 计算混合特征
            z_i = class_features(idx_i, :);
            z_j = class_features(idx_j, :);
            z_u = alpha * z_i + (1 - alpha) * z_j;
            
            % 存储混合特征及其标签
            mixed_features = [mixed_features; z_u];
            mixed_labels = [mixed_labels; unique_labels(c)];
        end
    end
    
    % 合并支持集和混合特征
    combined_features = [S_set; mixed_features];
    combined_labels = [S_set_label; mixed_labels];
    
    % 执行 k-means 聚类并迭代 3 次
    cluster_centers = zeros(num_classes, size(S_set, 2)); % 初始化簇中心
    max_iterations = 3; % 最大迭代次数
    
    % 随机初始化簇中心
    for c = 1:num_classes
        class_indices = find(combined_labels == unique_labels(c));
        % random_idx = class_indices(randi(length(class_indices)));
        % cluster_centers(c, :) = combined_features(random_idx, :);
        class_features = combined_features(class_indices, :);
        cluster_centers(c, :) = mean(class_features, 1); % 使用类别均值作为簇中心
    end
    
    for iter = 1:max_iterations
        % 计算样本到簇中心的余弦距离
        distances = zeros(size(combined_features, 1), num_classes);
        for c = 1:num_classes
            % for i = 1:size(combined_features, 1)
            %     % distances(i, c) = 1 - dot(combined_features(i, :), cluster_centers(c, :)) / ...
            %     %     (norm(combined_features(i, :)) * norm(cluster_centers(c, :)));
            % end
            distances(:, c) = vecnorm(combined_features - cluster_centers(c, :), 2, 2); % 欧几里得距离
        end
        
        % 计算软分配权重
        weights = exp(-distances); % 高斯核
        weights = weights ./ sum(weights, 2); % 归一化
        
        % 更新簇中心
        for c = 1:num_classes
            cluster_centers(c, :) = sum(weights(:, c) .* combined_features, 1) / sum(weights(:, c));
        end
    end
    
    % 对查询集进行预测
    predictions = zeros(size(Q_set, 1), 1);
    for q = 1:size(Q_set, 1)
        query_feature = Q_set(q, :);
        
        % 计算与每个簇中心的余弦距离
        distances = zeros(num_classes, 1);
        for c = 1:num_classes
            % distances(c) = 1 - dot(query_feature, cluster_centers(c, :)) / ...
            %     (norm(query_feature) * norm(cluster_centers(c, :)));
            distances(c) = norm(query_feature - cluster_centers(c, :), 2); % 欧几里得距离
        end
        
        % 使用公式 (4) 计算类概率
        scores = exp(-distances); % 计算分子
        probabilities = scores / sum(scores); % 归一化为概率
        
        % 预测概率最高的类别
        [~, max_idx] = max(probabilities);
        predictions(q) = unique_labels(max_idx);
    end
end
