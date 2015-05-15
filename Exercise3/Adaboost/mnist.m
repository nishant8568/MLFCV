% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
DATA_PATH = 'D:/Informatics/4th Semester/Lab Course/Exercises/Exercise 3/Dataset/';
images = loadMNISTImages(strcat(DATA_PATH,'train-images-idx3-ubyte'));
labels = loadMNISTLabels(strcat(DATA_PATH,'train-labels-idx1-ubyte'));
test_images = loadMNISTImages(strcat(DATA_PATH,'t10k-images-idx3-ubyte'));
test_labels = loadMNISTLabels(strcat(DATA_PATH,'t10k-labels-idx1-ubyte'));

images = images.';
images_part = images(1:1000, :);
% replace all 0's with 10
labels = changem(labels, 10, 0);
test_labels = changem(test_labels, 10, 0);
labels_part = labels(1:1000, :);
test_images = test_images.';
test_images_part = test_images(1:2000, :);
test_labels_part = test_labels(1:2000, :);

% checking for optimal number of iterations
%{
rounds = [100, 250, 500, 750, 1000];
complete_prediction = [];
matched_labels = [];

for j=1:5
        disp(rounds(j));
        % Training
        model = trainadaboost(images_part, labels_part, 10, rounds(j));
        % Testing
        disp('Testing.....');
        labels_predicted = zeros;
        for i = 1:2000
            label_predicted = classifyadaboost(test_images_part(i:i, :), model);
            %disp(label_predicted);
            labels_predicted(end+1) = label_predicted;
            if(rem(i, 100)==0)
                disp('i is : ');
                disp(i);
            end
            i=i+1;
        end
        matched = 0;
        for k = 1:2000
            if(labels_predicted(:, k+1:k+1)==test_labels_part(k:k, :))
                matched = matched+1;
            end
        end
        complete_prediction = [complete_prediction; labels_predicted];
        matched_labels = [matched_labels; matched];
    j=j+1;
end
complete_prediction = complete_prediction.';
display(complete_prediction)
%}

rounds = 500;
complete_prediction = [];
matched_labels = [];
disp(rounds);
% Training
model = trainadaboost(images, labels, 10, rounds);
% Testing
disp('Testing.....');
labels_predicted = zeros;
[m, n] = size(test_images);
for i = 1:m
    label_predicted = classifyadaboost(test_images(i:i, :), model);
    %disp(label_predicted);
    labels_predicted(end+1) = label_predicted;
    if(rem(i, 100)==0)
        disp('i is : ');
        disp(i);
    end
    i=i+1;
end
matched = 0;
matched_indices = [];
for k = 1:m
    if(labels_predicted(:, k+1:k+1)==test_labels(k:k, :))
        matched = matched+1;
        matched_indices(end+1) = k;
    end
end
complete_prediction = [complete_prediction; labels_predicted];
matched_labels = [matched_labels; matched];
complete_prediction = complete_prediction.';

% Evaluate classifier
[correctly_classified, wrongly_classified] = evaluate(test_images, matched_indices);