function[F, P, R, TP, FP, FN] = f_measure(GT, In)
%  
% input datasets are assumed to be binary 
%  

mask = In(GT==1);

% True positives
TP = length( mask(mask==1) );

% False positives
FP = In(GT==0);
FP = length( FP(FP == 1) );

% False negatives
FN = length( mask(mask == 0) );


% Precision
if( (TP==0) & (FN==0) )
  P = 1;        % case when GT is all zeros and there is no FN
else
  P = TP/(TP+FN);
end

% Recall
if( (TP==0) & (FP==0) )
  R = 1;        % case when GT is all zeros and there is no FP
else
  R = TP/(TP+FP);
end

% F-measure
F = (2*P*R)/(P+R);

