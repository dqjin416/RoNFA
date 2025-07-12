function [noised_data,noised_labels, noise_positions] = add_noise(features,labels, mask_indices, ways, noise_fraction, noise_type)
    % add_noise_only_labels: 仅在标签中添加噪声，保持特征数据不变
    %
    % Inputs:
    %   labels - 原始标签 (列向量)
    %   mask_indices - 包含支持集、噪声集的样本索引的结构
    %   ways - 类别数量
    %   noise_fraction - 噪声比例 (如 0.2 表示 20% 的标签被更改)
    %   noise_type - 噪声类型 (如 'sym_swap'、'pair_swap'、'outlier')
    %
    % Outputs:
    %   noised_labels - 添加噪声后的标签
    %   noise_positions - 噪声数据的位置（标记噪声样本的二进制数组）

    % 加载离群数据
    load('out_transformer_features.mat');  % 假设 out_feature 是一个变量，包含离群样本特征
    load('out_transformer_lables.mat');    % 假设 out_label 是一个变量，包含离群样本标签
    out_feature = out_transformer_features;
    out_label = out_transformer_lables;
    % [COEFF, SCORE, LATENT, TSQUARED] = pca((out_feature)) ;
    % out_feature=SCORE(:,1:4);

    % 计算支持集样本的数量及每类中的样本数量
    num_idx = mask_indices.support;  % 支持集的样本索引
    shot = int32(length(num_idx) / ways);  % 每类样本数量
    noise_num = round(noise_fraction * shot);  % 每类中添加噪声的数量

    % 选择需要添加噪声的标签索引
    indices_to_change = zeros(ways, noise_num, 'int16');
    for i = 1:ways
        class_i_idx = num_idx((i-1) * shot + 1 : i * shot);  % 当前类别的样本索引
        indices_to_change(i, :) = class_i_idx(randperm(shot, noise_num));
    end
    indices_to_change_flat = indices_to_change(:);  % 将多维索引展平为一维

    % 复制原始标签
    noised_labels = labels;
    noised_data = features;

    % 根据噪声类型修改标签
    switch noise_type
        case 'sym_swap'
            % 对称标签交换噪声 (随机选择标签)
            for c = 1:ways
                % swap_choices = randperm(ways);  % 生成类别交换顺序
                % swap_class = swap_choices(swap_choices ~= c);  % 排除当前类
                % noised_labels(indices_to_change(c, :)) = swap_class(1);  % 替换为新标签
                class_i_idx = num_idx((c-1) * shot + 1 : c * shot);  % 当前类别的样本索引
                
                % 选择需要添加噪声的标签索引
                indices_to_change = class_i_idx(randperm(shot, noise_num));  % 随机选择需要添加噪声的样本
                
                % 选择噪声标签（确保噪声标签不超过干净类）
                valid_noise_classes = setdiff(1:ways, c);  % 当前类的有效噪声类
                noise_labels = valid_noise_classes(randperm(length(valid_noise_classes), noise_num));
                
                % 确保噪声标签数量不超过干净类的限制
                for n = 1:noise_num
                    noise_class = noise_labels(n);
                    class_count = sum(noised_labels(class_i_idx) == noise_class);
                    max_allowed = shot * (1 - noise_fraction);  % 允许的最大噪声数量
                    
                    if class_count < max_allowed
                        noised_labels(indices_to_change(n)) = noise_class;  % 替换为新标签
                    else
                        % 随机选择其他噪声标签
                        valid_choices = setdiff(valid_noise_classes, noise_class);
                        if ~isempty(valid_choices)
                            new_noise_class = valid_choices(randi(length(valid_choices)));
                            noised_labels(indices_to_change(n)) = new_noise_class;  % 替换为新标签
                        end
                    end
                end
            end

        case 'pair_swap'
            % 配对标签交换噪声 (在每类间配对交换)
            deranged_classes = gen_derangement(ways);  % 生成一个错位排列
            for c = 1:ways
                noised_labels(indices_to_change(c, :)) = deranged_classes(c);  % 标签替换为配对类别
            end

        case 'outlier'
            % 对称标签交换噪声 (随机选择标签)
            OUTLIER = -1;  % 定义离群标记
            for c = 1:ways
                class_i_idx = num_idx((c-1) * shot + 1 : c * shot);  % 当前类别的样本索引
                
                % 选择需要添加噪声的标签索引
                indices_to_change = class_i_idx(randperm(shot, noise_num));  % 随机选择需要添加噪声的样本
                
                % 选择噪声标签（从 out_label 中随机选择）
                noise_labels = out_label(randperm(length(out_label), noise_num));  % 随机选择噪声标签
                
                % 确保噪声标签数量不超过干净类的限制
                for n = 1:noise_num
                    noise_class = noise_labels(n);
                    class_count = sum(noised_labels(class_i_idx) == noise_class);
                    max_allowed = shot * (1 - noise_fraction);  % 允许的最大噪声数量
                    
                    if class_count < max_allowed
                        % 替换为新标签，并将噪声标签设置为离群标记
                        noised_labels(indices_to_change(n)) = OUTLIER;  % 将标签设置为离群标记
                        
                        % 替换对应特征为离群特征
                        outlier_feature_index = randi(size(out_feature, 1));  % 随机选择离群特征的索引
                        noised_data(indices_to_change(n), :) = out_feature(outlier_feature_index, :);  % 替换特征
                    else
                        % 随机选择其他噪声标签
                        valid_choices = setdiff(out_label, noise_class);  % 排除当前噪声类
                        if ~isempty(valid_choices)
                            new_noise_class = valid_choices(randi(length(valid_choices)));
                            noised_labels(indices_to_change(n)) = OUTLIER;  % 将标签设置为离群标记
                            
                            % 替换对应特征为离群特征
                            outlier_feature_index = randi(size(out_feature, 1));  % 随机选择离群特征的索引
                            noised_data(indices_to_change(n), :) = out_feature(outlier_feature_index, :);  % 替换特征
                        end
                    end
                end
            end
        otherwise
            error('未实现的噪声类型');
    end

    % 记录噪声位置 (二进制标记)
    noise_positions = zeros(size(labels));
    noise_positions(indices_to_change_flat) = 1;  % 将噪声样本位置标记为 1

end

function derangement = gen_derangement(n)
    % gen_derangement: 生成没有固定点的排列
    derangement = randperm(n);
    while any(derangement == (1:n))
        derangement = randperm(n);
    end
end