%load fisheriris 
%Mdl1 = fitcsvm(X,Y);
%%
[X, Y] = generate_data('twinpeaks', 2000);
rng(10); % For reproducibility	
figure, scatter3(X(:,1), X(:,2), X(:,3), 5, Y); title('Original dataset'), drawnow
	
no_dims = round(intrinsic_dim(X, 'MLE'));
	
disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);
	
[mappedX, mapping] = compute_mapping(X, 'LLE', no_dims);	
	
figure, scatter(mappedX(:,1), mappedX(:,2), 5, Y); title('Result of LDA');
    
[mappedX, mapping] = compute_mapping(X, 'Laplacian', no_dims, 7);	
	
figure, scatter(mappedX(:,1), mappedX(:,2), 5, Y(mapping.conn_comp)); title('Result of Laplacian Eigenmaps'); drawnow
%%
Mdl = fitcsvm(X,Y,'KernelFunction','linear','Standardize',true);
rloss = resubLoss(Mdl)
CVMdl = crossval(Mdl);
kloss = kfoldLoss(CVMdl)