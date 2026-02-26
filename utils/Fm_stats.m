function[Fmean Fmed Fmode Fcum] = Fm_stats(binaryfIndex, incStats)


[h, bins] = hist( incStats.Fmeasure(binaryfIndex(:,2)), 20 );
[dummy, pos] = max(h);
Fmode = bins(pos);

%  Fmean = mean(  incStats.Fmeasure(myPars.binaryfIndex(1,2):end) );
%  Fmed  = median(  incStats.Fmeasure(myPars.binaryfIndex(1,2):end) );
Fmean = mean(  incStats.Fmeasure(binaryfIndex(:,2)) );
Fmed  = median(  incStats.Fmeasure(binaryfIndex(:,2)) );



TPcum = incStats.TP(binaryfIndex(:,2));
FPcum = incStats.FP(binaryfIndex(:,2));
FNcum = incStats.FN(binaryfIndex(:,2));

Pcum = TPcum/(TPcum+FNcum);
Rcum = TPcum/(TPcum+FPcum);
Fcum = (2*Pcum*Rcum)/(Pcum+Rcum);
