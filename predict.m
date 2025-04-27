function [y_pred] = predict(X, weight1, bias1, weight2, bias2)
 % Forward pass for prediction
 z1 = X * weight1 + repmat(bias1, size(X, 1), 1);
 h1 = 1 ./ (1 + exp(-z1));
 z2 = h1 * weight2 + repmat(bias2, size(h1, 1), 1);
 h2 = 1 ./ (1 + exp(-z2));
 [~, y_pred] = max(h2, [], 2);
end