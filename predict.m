function y_pred = predict(X, weight1, bias1, weight2, bias2)
% Predict output based on trained weights
h1 = log_sig(X * weight1 + bias1);
h2 = log_sig(h1 * weight2 + bias2);
y_pred = h2 > 0.5; % binary classification: 0 or 1
end
