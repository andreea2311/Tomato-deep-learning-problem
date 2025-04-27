function [] = evaluate(y_true, y_pred)
 % Convert one-hot encoded y_true back to class indices
 [~, y_true_idx] = max(y_true, [], 2);
 
 % Calculate accuracy
 accuracy = mean(y_pred == y_true_idx) * 100;
 fprintf('Accuracy: %.2f%%\n', accuracy);
 
 % Calculate confusion matrix
 num_classes = size(y_true, 2);
 conf_matrix = zeros(num_classes, num_classes);
 for i = 1:length(y_true_idx)
     conf_matrix(y_true_idx(i), y_pred(i)) = conf_matrix(y_true_idx(i), y_pred(i)) + 1;
 end
 
 % Display confusion matrix
 fprintf('Confusion Matrix:\n');
 disp(conf_matrix);
end