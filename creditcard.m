
%X = raw credit card data
%numf = number of features
%N = number of samples

X = load('creditcard.mat'); %Read in cvs data %Columns: Time between transactions, 28 nameless features, amount, class
X = X.creditcard;
numf = size(X,2); %Number of features
X = sortrows(X,numf); %Put positives at the end of data set
N = size(X,1); %Number of samples in total

strippedX = X(:,[2:29 31]); %Strip off amount, time features
numf = numf-2; %Features drop by 2
numpos = sum(strippedX(:,numf)); %Find out how many positives we have in the data set. (29th feature is class).


%Normalize features (except for classification)
strippedX(:,1:end-1) = strippedX(:,1:end-1) - mean(strippedX(:,1:end-1));
strippedX(:,1:end-1) = strippedX(:,1:end-1)./std(strippedX(:,1:end-1));


%Put 50% of all negs and pos into training set, 20% into testset, 30% into
%validation set
rndindicesneg = randperm(N-numpos); %random indices for choosing negative samples
rndindicespos = randperm(numpos) + (N-numpos); %random indices for choosing positive samples

training_indices_neg = rndindicesneg(1:floor(0.5*size(rndindicesneg,2))); %random 50% negative samples go here
validation_indices_neg = rndindicesneg(floor(0.5*size(rndindicesneg,2))+1:floor(0.8*size(rndindicesneg,2))); %Next 30% go here
test_indices_neg = rndindicesneg(floor(0.8*size(rndindicesneg,2))+1:end); %Rest go here

training_indices_pos = rndindicespos(1:floor(0.5*size(rndindicespos,2))); %random 80% positive samples go here
validation_indices_pos = rndindicespos(floor(0.5*size(rndindicespos,2))+1:floor(0.8*size(rndindicespos,2))); %Next 30% go here
test_indices_pos = rndindicespos(floor(0.8*size(rndindicespos,2))+1:end); %Rest go here

trainingset = [strippedX(training_indices_neg,:); strippedX(training_indices_pos,:)]; %Construct training set
validationset = [strippedX(validation_indices_neg,:); strippedX(validation_indices_pos,:)]; %Construct training set
testset = [strippedX(test_indices_neg,:); strippedX(test_indices_pos,:)]; %Construct test set

%Shuffle training and test sets again
trainingset = trainingset(randperm(size(trainingset,1)),:);
validationset = validationset(randperm(size(validationset,1)),:);
testset = testset(randperm(size(testset,1)),:);

y = trainingset(:,numf); %labels for training set
trainingset = trainingset(:,1:numf-1); %Remove labels from dataset
vy = validationset(:,numf); %labels for validation set
validationset = validationset(:,1:numf-1); %Remove labels from dataset
testy = testset(:,numf); %labels for test set
testset = testset(:,1:numf-1); %Remove labels from dataset

% %Try undersampling training and test sets

%Put 80% of all pos into training set, 20% into testset. Match num of negative samples to positive 
urndindicesneg = randperm(N-numpos,numpos); %random indices for choosing negative samples
urndindicespos = randperm(numpos) + (N-numpos); %random indices for choosing positive samples

utraining_indices_neg = urndindicesneg(1:floor(0.5*size(urndindicesneg,2))); %random 80% negative samples go here
uvalidation_indices_neg = urndindicesneg(floor(0.5*size(urndindicesneg,2))+1:floor(0.8*size(urndindicesneg,2))); %Next 30% go here
utest_indices_neg = urndindicesneg(floor(0.8*size(urndindicesneg,2))+1:end); %Rest go here

utraining_indices_pos = urndindicespos(1:floor(0.5*size(urndindicespos,2))); %random 80% positive samples go here
uvalidation_indices_pos = urndindicespos(floor(0.5*size(urndindicespos,2))+1:floor(0.8*size(urndindicespos,2))); %Next 30% go here
utest_indices_pos = urndindicespos(floor(0.8*size(urndindicespos,2))+1:end); %Rest go here

utrainingset = [strippedX(utraining_indices_neg,:); strippedX(utraining_indices_pos,:)]; %Construct training set
uvalidationset = [strippedX(uvalidation_indices_neg,:); strippedX(uvalidation_indices_pos,:)]; %Construct training set
utestset = [strippedX(utest_indices_neg,:); strippedX(utest_indices_pos,:)]; %Construct test set

%Shuffle training and test sets again
utrainingset = utrainingset(randperm(size(utrainingset,1)),:);
uvalidationset = uvalidationset(randperm(size(uvalidationset,1)),:);
utestset = utestset(randperm(size(utestset,1)),:);

uy = utrainingset(:,numf); %labels
utrainingset = utrainingset(:,1:numf-1); %Remove labels from dataset
uvy = uvalidationset(:,numf); %labels for validation set
uvalidationset = uvalidationset(:,1:numf-1); %Remove labels from dataset
utesty = utestset(:,numf); %labels
utestset = utestset(:,1:numf-1); %Remove labels from dataset




% Initialize fitting parameters
initial_theta = zeros(numf-1, 1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 800);

%Find optimal lambda on undersampled validation set
lam = [0.01 0.1 1 10 50 100 200 300 500 1000];
mincost = inf;
mini = 1;
for i = 1: size(lam,2)
    [theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, utrainingset, uy, lam(i))), initial_theta, options);
    if costFunctionReg(theta, uvalidationset, uvy, 0) < mincost
        mincost = costFunctionReg(theta,uvalidationset,uvy,0);
        mini = i;
    end
end

lambda = lam(i); %Set lambda to optimal value

%Train using optimal lambda on full training set
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, trainingset, y, lambda)), theta, options);


%Predict on test set
p = predict(theta, testset);

%True positives
Tp = sum(p.*testy)
%False positiives
Fp = sum(p.*~testy)
%False negatives
Fn = sum(~p.*testy);

fprintf('Precision: %f\n', Tp/(Tp+Fp)); 
fprintf('Recall: %f\n', Tp/(Tp+Fn));


% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, utrainingset, uy, lambda)), initial_theta, options);

%Predict undersampled test set
p = predict(theta, testset); %Use stricter classifier

%True positives
Tp = sum(p.*testy)
%False positiives
Fp = sum(p.*~testy)
%False negatives
Fn = sum(~p.*testy);

fprintf('Precision: %f\n', Tp/(Tp+Fp)); 
fprintf('Recall: %f\n', Tp/(Tp+Fn));
