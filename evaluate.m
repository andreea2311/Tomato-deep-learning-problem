function evaluate(y_true, y_pred)
    % Calculate accuracy
    accuracy = mean(y_true == y_pred);
    
    % Calculate precision, recall, and F1 score
    tp = sum(y_true == 1 & y_pred == 1);
    fp = sum(y_true == 0 & y_pred == 1);
    fn = sum(y_true == 1 & y_pred == 0);
    
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1_score = 2 * (precision * recall) / (precision + recall);
    
    % Display metrics
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Precision: %.4f\n', precision);
    fprintf('Recall: %.4f\n', recall);
    fprintf('F1 Score: %.4f\n', f1_score);
    
    % Confusion matrix
    fprintf('Confusion Matrix:\n');
    fprintf('TP: %d, FP: %d\n', tp, fp);
    fprintf('FN: %d, TN: %d\n', fn, sum(y_true == 0 & y_pred == 0));
end