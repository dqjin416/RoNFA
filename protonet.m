function [distances, loss, acc] = protonet(support_features, support_labels, query_features, query_labels, nways, k_shot)
    % Protonet 实现：计算距离、交叉熵损失和准确率
    %
    % 参数：
    % support_features - 支持集的特征 [num_support_samples, feature_dim]
    % support_labels - 支持集的标签 [num_support_samples, 1]
    % query_features - 查询集的特征 [num_query_samples, feature_dim]
    % query_labels - 查询集的真实标签 [num_query_samples, 1]
    % nways - 类别数量
    % k_shot - 每类样本数量
    %
    % 返回：
    % distances - 查询样本到原型的欧几里得距离
    % loss - 交叉熵损失
    % acc - 查询集预测的准确率

    % 计算原型
    prototypes = zeros(nways, size(support_features, 2)); % 初始化原型矩阵
    for i = 1:nways
        % 获取当前类别的支持集样本
        class_mask = (support_labels == i); % 假设标签从 1 开始
        class_features = support_features(class_mask, :);
        
        % 直接计算当前类别所有样本的原型
        prototypes(i, :) = mean(class_features, 1); % 计算所有样本的平均值作为原型
    end
    
    % 计算查询样本到原型的欧几里得距离
    distances = euclidean_distance(query_features, prototypes);
    
    % 计算交叉熵损失和准确率
    query_labels = query_labels(:); % 确保标签为列向量
    logits = -distances; % 取负值作为 logits
    logits_exp = exp(logits);
    probs = logits_exp ./ sum(logits_exp, 2); % 计算 softmax 概率
    
    % 真实类别的概率
    num_queries = length(query_labels);
    true_probs = arrayfun(@(i) probs(i, query_labels(i)), 1:num_queries); % MATLAB 索引从 1 开始
    
    % 计算负对数概率损失
    log_probs = -log(true_probs);
    loss = mean(log_probs);
    
    % 预测标签和准确率
    [~, preds] = max(logits, [], 2); % 按列取最大值，返回索引
    acc = mean(preds == query_labels); % MATLAB 索引从 1 开始
end

function dist = euclidean_distance(query, prototypes)
    % 欧几里得距离计算
    num_queries = size(query, 1);
    num_prototypes = size(prototypes, 1);
    dist = zeros(num_queries, num_prototypes);
    for i = 1:num_queries
        for j = 1:num_prototypes
            dist(i, j) = sqrt(sum((query(i, :) - prototypes(j, :)) .^ 2));
        end
    end
end
