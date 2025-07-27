% Parameters
nways = 5;          % Number of classes to select
sample_num = 20;    % Number of samples per class

% Select nways unique classes randomly from available classes
available_classes = unique(labels);
selected_classes = randperm(length(available_classes), nways);
class = available_classes(selected_classes);

% Initialize data containers
data = zeros(sample_num*nways, size(features, 2));
data_label = zeros(sample_num*nways, 1);

% For each selected class
for i = 1:nways
    % Get all samples for this class
    class_samples = features(labels == class(i), :);
    class_labels = labels(labels == class(i));
    
    % Randomly select sample_num samples without replacement
    total_samples = size(class_samples, 1);
    if total_samples < sample_num
        error('Not enough samples in class %d. Required: %d, Available: %d', ...
              class(i), sample_num, total_samples);
    end
    
    selected_indices = randperm(total_samples, sample_num);
    selected_samples = class_samples(selected_indices, :);
    
    % Store the selected samples
    start_idx = (i-1)*sample_num + 1;
    end_idx = i*sample_num;
    
    data(start_idx:end_idx, :) = selected_samples;
    data_label(start_idx:end_idx) = i;  % Using sequential labels 1:nways
end

% Optional: Shuffle the dataset if needed
shuffle_idx = randperm(size(data, 1));
instance = data(shuffle_idx, :);
instance_label = data_label(shuffle_idx);