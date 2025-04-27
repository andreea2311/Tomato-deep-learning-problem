imageFolder = 'tomato_data'; % Folder with resized images
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split into training and testing
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

numTrain = numel(imdsTrain.Files);
numTest = numel(imdsTest.Files);
imageSize = [64, 64, 3]; % height, width, channels
XTrain = zeros(numTrain, prod(imageSize), 'double');
XTest = zeros(numTest, prod(imageSize), 'double');

% Load training data
for i = 1:numTrain
    img = im2double(readimage(imdsTrain, i)); % Read and convert to double
    XTrain(i, :) = img(:)';
end
yTrain = imdsTrain.Labels;

% Load testing data
for i = 1:numTest
    img = im2double(readimage(imdsTest, i));
    XTest(i, :) = img(:)';
end
yTest = imdsTest.Labels;

% Normalize features
meanX = mean(XTrain, 1);
stdX = std(XTrain, [], 1);
XTrain = (XTrain - meanX) ./ (stdX + 1e-8); % add small epsilon to avoid div-by-zero
XTest = (XTest - meanX) ./ (stdX + 1e-8);

% Convert labels to one-hot encoding
classNames = categories(yTrain);
numClasses = numel(classNames);
YTrain = onehotencode(yTrain, 2);
YTest = onehotencode(yTest, 2);

% Convert data types
X_train = double(XTrain);
X_test = double(XTest);
Y_train = double(YTrain);
Y_test = double(YTest);
