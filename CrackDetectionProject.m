%% Main Function for Crack Detection using Supervised Machine Learning Algorithms (Support Vector Machines and Decision Trees)
function CrackDetectionProject

%Define Folder containing input images
inputImageFolder = 'C:\Users\pc\Documents\DIGITAL ENGINEERING\Summer 2024\Image Analysis and Object Recognition\Exercise\Final Project\Bunmi\Crack-Detector\data\Original Image'; %Modify as suitable

%Define folder cointaining annotated images done with GIMP software
annotedImageFolder = 'C:\Users\pc\Documents\DIGITAL ENGINEERING\Summer 2024\Image Analysis and Object Recognition\Exercise\Final Project\Bunmi\Crack-Detector\data\Image Annotation'; %Modify as suitable

%Listing alll image files in the raw input folder
inputImageFiles = dir(fullfile(inputImageFolder, '*.jpg'));

%Defining the result image folder parts
resultImageFolder = 'C:\Users\pc\Documents\DIGITAL ENGINEERING\Summer 2024\Image Analysis and Object Recognition\Exercise\Final Project\Bunmi\Crack-Detector\data\Results'; %Modify as suitable


%Proceeding to Process all Images
for imgIdx = 1:length(inputImageFiles)

    %Reading the input image name from the input folder
    inputImageName = fullfile(inputImageFolder, inputImageFiles(imgIdx).name);
    inputImage = imread(inputImageName);

    %Defining the annotated image file name based on the input image
    [~, baseFilename, ext] = fileparts(inputImageFiles(imgIdx).name);
    annotatedImageName = fullfile(annotedImageFolder, [baseFilename, '_annotated', ext]);
    annotatedImage = imread(annotatedImageName);

    %Performing Adaptive Thresholding to binarize Images
    [greyImage, binaryImage] = AdaptiveThresholding(inputImage);

    %Plotting and Visualizing the original Image
    figure;
    imshow(inputImage);
    title(['Original image ', num2str(imgIdx)]);

    %Plotting and Visualizing the Histogram of Images
    histogramFigure = figure;
    imhist(greyImage);
    title(['Original image histogram ', num2str(imgIdx)]);

    %Save grey scale image in result folder
    grayImageFileName = fullfile(resultImageFolder, ['Gray_Image_', num2str(imgIdx), '.png']);
    imwrite(greyImage, grayImageFileName);

    %Saving Histogram Plots to results folder
    histogramFileName = fullfile(resultImageFolder, ['Histogram_Plot_', num2str(imgIdx), '.png']);
    saveas(histogramFigure, histogramFileName);
    close(histogramFigure);

    %Performing Morphological Operation on Image
    [thinnedCrackImage, segmentedCrackImage] = MorphologicalOperators(binaryImage);

    %Saving the thinned image to the results folder
    morhologicalImageFileName = fullfile(resultImageFolder, ['Morphological_Image_', num2str(imgIdx), '.png']);
    imwrite(segmentedCrackImage, morhologicalImageFileName);

    %Visulaizing the thinned cracked images
    figure;
    imshow(thinnedCrackImage);
    title(['Thinned Cracks - Image ', num2str(imgIdx)]);

    %Saving the thinned Image to Result folder
    thinnedImageFileName = fullfile(resultImageFolder, ['Thinned_Image_', num2str(imgIdx), '.png']);
    imwrite(thinnedCrackImage, thinnedImageFileName);

    %Defining the binary mask of the segmented cracks
    binaryMask = segmentedCrackImage;

    %Peform Connected Component Labeling to Extract features
    [features, labels] = ConnectedComponentLabeling(annotatedImage, binaryMask);

end

%Calculating the amount of crack and non-crack regions
crackCount = 0;
nonCrackCount = 0;
for idx = 1:size(labels)
    if labels(idx) == 1
        crackCount = crackCount + 1;
    else
        nonCrackCount = nonCrackCount + 1;
    end
end

%Display the amount of crack and non crack region in Images
disp(['Number of Crack region: ', num2str(crackCount)]);
disp(['Number of Non Crack region: ', num2str(nonCrackCount)]);

%Splitting the Datasets into test and training sets
rng(1); %Setting a random seed for reproducibility
numofSamples = size(features, 1);
splitRatio = 0.8; % 80% of Image for training and 20% for testing
index = randperm(numofSamples);
splitIndex = round(splitRatio * numofSamples);

%splitting the Data
trainFeatures = features(index(1:splitIndex), :); %Randomizing the training features
testFeatures = features(index(splitIndex+1:end), :); %Randomizing the test Features

trainLabels = labels(index(1:splitIndex));
testLabels = labels(index(splitIndex + 1: end));

%Classifying the Training Image samples using SVM and Decision Tree
[svmClassifier, decisionTreeClassifier] = SVMandDecisionTreeClassifier(trainFeatures, trainLabels);

%Predicting Test Data labels
[predictedLabelsSVM, predictedLabelDTree] = PredictTestDataLabel(testFeatures, svmClassifier, decisionTreeClassifier);

%Calculating the Crack length in test images
CalculateCrackLength(testFeatures, predictedLabelsSVM);

% Calculating the classifier's performance
accuracy = sum(predictedLabelsSVM == testLabels) / numel(testLabels);
disp(['Accuracy of SVM Classifier: ', num2str(accuracy * 100), '%']);

accuracy = sum(predictedLabelDTree == testLabels) / numel(testLabels);
disp(['Accuracy of Decision Tree Classifier: ', num2str(accuracy*100),'%']);

end

%% Implementing a Function to Binarize Image
function [greyImage, binaryImage] = AdaptiveThresholding(image)

    %Convert Image to grey scale
    greyImage = rgb2gray(image);

    %perform adaptive thresholding
    binaryImage = imbinarize(greyImage, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.49);

    %Filling up small holes in binarized images
    binaryImage = imfill(binaryImage, "holes");

    %Cleaning up Binary image removing connected components
    binaryImage = bwareaopen(binaryImage, 100);

end

%% Implementing Morphological operators to Clean Binarized Image
function [thinnedCrackImage, segmentedCrackImage] = MorphologicalOperators(binaryImage)

    % Inverting the binary mask to make cracks white and no-crack black
    segmentedCrackImage = ~binaryImage;

    %defining a structuring element and performing morphological Closing
    se = strel('disk', 3); %Adjust as needed
    segmentedCrackImage = imclose(segmentedCrackImage, se);

    %Defining a structuring element and performing Morphological Opening
    se = strel('disk', 2); %Adjust as needed
    segmentedCrackImage = imopen(segmentedCrackImage, se);

    %Performing Morphological Thinning
    thinnedCrackImage = bwmorph(segmentedCrackImage, 'thin', 10);
end

%% Implementing a function to Perform Connected Component labeling and Extract Features from Image Content
function [features, labels] = ConnectedComponentLabeling(annotatedImage, binaryMask)

    %Initializing an array to collect features and labels
    features = [];
    labels = [];

    %Perform connected component labeling using bwlable
    [labelMatrix, numOfRegions] = bwlabel(binaryMask);

    %Extracting unique labels as regions
    uniqueLabels = unique(labelMatrix);

    %Extract features and assign labels based on annotated image
    for regionIdx = 1:numOfRegions
        regionMask = (labelMatrix == uniqueLabels(regionIdx));

        % Calculating region properties using regionprops on regionMask
        % source: https://de.mathworks.com/help/images/ref/regionprops.html
        stats = regionprops(regionMask, 'Area', 'Perimeter', 'Eccentricity'); %Area is number of pixels

        %Calculating the label based on annotated image
        annotationValue = mean(annotatedImage(regionMask));

        if annotationValue > 100
            label = 1; %Crack Region
        else
            label = 0; %Non-Crack Region
        end

        %Extracting features from 'stats' and assigning labels
        area = stats.Area;
        perimeter = stats.Perimeter;
        eccentricity = stats.Eccentricity;

        %Defining feature vector over this region
        featureVector = [area, perimeter, eccentricity]; %Add more features here if needed

        %Storing the feature labels and vectors respectively
        features = [features; featureVector];
        labels = [labels; label];
    end
end

%% Implementing a training classifier using Support Vector Machine (SVM) and Decision Tree
function [svmClassifier, decisionTreeClassifier] = SVMandDecisionTreeClassifier(trainFeatures, trainLabels)

% Training an SVM classifier
% Source: https://www.mathworks.com/help/stats/fitcsvm.html
svmClassifier = fitcsvm(trainFeatures, trainLabels);

% Training an Decision Tree classifier
% Source: https://www.mathworks.com/help/stats/fitctree.html
decisionTreeClassifier = fitctree(trainFeatures, trainLabels);

end

%% Implementing a Function to predict label for test data
function [predictedLabelsSVM, predictedLabelDTree] = PredictTestDataLabel(testFeatures, svmClassifier, decisionTreeClassifier)

predictedLabelsSVM = predict(svmClassifier, testFeatures);
predictedLabelDTree = predict(decisionTreeClassifier, testFeatures);

end

%% Implementing a Function to Calculate Crack length
function CalculateCrackLength(testFeatures, predictedLabelsSVM)

for count = 1: size(predictedLabelsSVM)
    if predictedLabelsSVM(count) == 1
        crackLength = testFeatures(count);
        disp(['Crack length of the region is : ', num2str(crackLength), ' pixels']);
    end
end
end