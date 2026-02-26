function[] = compute_Fm(GTdir, InDir, fIndex)
%  
% It is assumed that 
%  
%  * InDir has the results via 'segmentation method' X; such results are
%    N binary images ('mat' format; names are 'nameBase' follow by a correlative number, 
%    e.g. file001 file002 ... file'N'
%  
%  * GTdir hast the ground-truth segmented files; M (M<=N) binary imagez are given
%    names are 'nameBase' follow by a correlative number, e.g. gt001, gt002, ...
%    gt'M'.
%  
%  * fIndex: vector that gives the correspondance between files in GTdir and InDir,
%    i.e. if fIndex = [1, 10; 2, 11; 3, 20] then the correspondance is gt001 -- file010,
%    gt002 -- file011 and gt003 -- file020
%  



