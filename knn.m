%load fisheriris %load data
[X, Y] = generate_data('twinpeaks', 10000);
rng(10); % For reproducibility	
figure, scatter3(X(:,1), X(:,2), X(:,3), 5, Y); title('Original dataset'), drawnow
	
no_dims = round(intrinsic_dim(X, 'MLE'));
disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);
[mappedX, mapping] = compute_mapping(X, 'PCA', no_dims);	
%figure, scatter(mappedX(:,1), mappedX(:,2), 5, Y); title('Result of PCA');
%[mappedX, mapping] = compute_mapping(X, 'Laplacian', no_dims, 7);	
%figure, scatter(mappedX(:,1), mappedX(:,2), 5, Y(mapping.conn_comp)); title('Result of Laplacian Eigenmaps'); drawnow


Mdl = fitcknn(X,Y,'NumNeighbors',4); %KNN CLassifier
rloss = resubLoss(Mdl)
CVMdl = crossval(Mdl);
kloss = kfoldLoss(CVMdl)




%%
%X = meas;
%Y = species;
%rng(10); % For reproducibility
%Mdl = fitcknn(X,Y,'NumNeighbors',4); %KNN CLassifier

%rloss = resubLoss(Mdl) %L = resubLoss(ens) returns the resubstitution loss, meaning the loss computed for the data that fitensemble used to create ens.

%CVMdl = crossval(Mdl);
%kloss = kfoldLoss(CVMdl)

