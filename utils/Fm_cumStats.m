function[Fmean Fmed Fcum] = Fm_stats(binaryfIndex, incStats)

if 0
Fmean = mean(  incStats.Fmeasure(binaryfIndex(:,2)) );
Fmed  = median(  incStats.Fmeasure(binaryfIndex(:,2)) );
else
Fmean = 0;
Fmed  = 0;
end


TPcum = sum( incStats.TP(binaryfIndex(:,2)) );
FPcum = sum( incStats.FP(binaryfIndex(:,2)) );
FNcum = sum( incStats.FN(binaryfIndex(:,2)) );

%  TPcum = sum( incStats.TPk(binaryfIndex(:,2)) );
%  FPcum = sum( incStats.FPk(binaryfIndex(:,2)) );
%  FNcum = sum( incStats.FNk(binaryfIndex(:,2)) );


Pcum = TPcum/(TPcum+FNcum);
Rcum = TPcum/(TPcum+FPcum);
Fcum = (2*Pcum*Rcum)/(Pcum+Rcum);
