load DataSet %load data


%%
X = meas;
Y = species;
rng(10); % For reproducibility
Mdl = fitcknn(X,Y,'NumNeighbors',4); %KNN CLassifier

rloss = resubLoss(Mdl) %L = resubLoss(ens) returns the resubstitution loss, meaning the loss computed for the data that fitensemble used to create ens.

CVMdl = crossval(Mdl);
kloss = kfoldLoss(CVMdl)


