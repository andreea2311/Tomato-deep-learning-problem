% Set the correct path for the image folder
imageFolder = 'D:/facultate/tema_optimizari/tomato/test/Tomato___Bacterial_spot';  % Update with your actual path

outputFolder = 'tomato_data/test/bacterial_spot';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Create the image datastore
imds = imageDatastore(imageFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Get the number of files in the datastore
numFiles = numel(imds.Files);

% Loop through the datastore and save resized images
for i = 1:numFiles
    % Read the current image using the i-th file
    img = readimage(imds, i);  % Read the i-th image
    
    % Resize the image to 64x64
    resized_img = imresize(img, [64, 64]);
    
    % Get the current file path
    currentFilePath = imds.Files{i};
    
    % Extract the filename without path and extension
    [~, name, ext] = fileparts(currentFilePath);
    
    % Create a unique output filename (add an index to avoid overwriting)
    outputFileName = fullfile(outputFolder, sprintf('%s_resized_%d%s', name, i, ext));
    
    % Save the resized image with the unique filename
    imwrite(resized_img, outputFileName);
end

fprintf('Images resized and saved successfully.\n');
