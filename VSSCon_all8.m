function [VSS, VSS_Y] = VSSCon_all8(S_set, S_set_label)
    % 参数初始化
    alpha = 0.8; % 混合特征比例
    num_mixed_features = 0; % 每类生成的混合特征数量
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
        
        % 生成不重复的混合特征
        for k = 1:num_mixed_features
            % 随机选择两个不同的样本
            idx_i = randi(num_samples);
            idx_j = randi(num_samples);
            while idx_j == idx_i
                idx_j = randi(num_samples);
            end
            
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
    
    % 初始化 k-means 簇中心
    cluster_centers = zeros(num_classes, size(S_set, 2)); % 初始化簇中心
    for c = 1:num_classes
        class_indices = find(combined_labels == unique_labels(c));
        class_features = combined_features(class_indices, :);
        cluster_centers(c, :) = mean(class_features, 1); % 使用类别均值初始化
    end
    
    % k-means 软聚类迭代
    max_iterations = 1; % 最大迭代次数
    for iter = 1:max_iterations
        % 计算样本到簇中心的欧几里得距离
        distances = zeros(size(combined_features, 1), num_classes);
        for c = 1:num_classes
            distances(:, c) = vecnorm(combined_features - cluster_centers(c, :), 2, 2);
        end
        
        % 计算软分配权重
        weights = exp(-distances); % 高斯核
        weights = weights ./ sum(weights, 2); % 归一化
        
        % 更新簇中心
        for c = 1:num_classes
            cluster_centers(c, :) = sum(weights(:, c) .* combined_features, 1) / sum(weights(:, c));
        end
    end
    
    % 计算加权平均作为代表样本
    VSS = zeros(num_classes, size(S_set, 2)); % 初始化代表样本
    for c = 1:num_classes
        % 获取当前类别的样本索引
        class_indices = find(combined_labels == unique_labels(c));
        class_features = combined_features(class_indices, :);
        % num_samples = size(class_features , 1);
        % % 计算样本间的距离矩阵
        % distance_matrix = zeros(num_samples, num_samples);
        % for m = 1:num_samples
        %     for n = 1:num_samples
        %         if m ~= n
        %             % 计算欧几里得距离
        %             % distance_matrix(m, n) = norm(class_features(m, :) - class_features(n, :));
        %             % 计算余弦相似度
        %             cosine_similarity = dot(class_features (m, :), class_features (n, :)) / ...
        %                 (norm(class_features (m, :)) * norm(class_features (n, :)));
        %             distance_matrix(m, n) = 1 - cosine_similarity; % 转换为距离（1 - cosine similarity）
        %         end
        %     end
        % end
        % 
        % % 计算相似性分数 a_i^(c)
        % similarity_scores = -mean(distance_matrix, 2); % 平均距离的负值
        % 
        % % 计算权重 w_i^(c)
        % T = 0.4; % 温度参数
        % weights = exp(similarity_scores / T);
        % weights = weights / sum(weights); % 归一化权重
        % 
        % % 根据权重计算代表样本
        % VSS(c, :) = weights' * class_features ; % 加权平均得到代表样本
        % % 直接计算均值
        % VSS(c, :) = mean(class_features, 1);
        % 获取该类别对应的软分配权重
        class_weights = weights(class_indices, c);
        class_weights = class_weights / sum(class_weights); % 归一化

        % 计算加权平均
        VSS(c, :) = sum(class_weights .* class_features, 1);
    end
    
    % 设置输出
    VSS_Y = unique_labels; % 簇的标签与类别一致
end
