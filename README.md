# IncPCP-PTI

IncPCP-PTI is a  fully incremental algorithm for video background modeling that extends incPCP-TI in order to cope with moving and panning cameras. The method continuously updates the low-rank component in order to align it to the current reference frame of the camera.
In this page, we include the Matlab code and results on a synthetic panning and jitter video along with results on real videos from the CDNet 2014 dataset associated with the paper G. Chau and P. Rodriguez, "Panning and Jitter Invariant Incremental Principal Component Pursuit for Video Background Modeling" submitted to the 2nd "International Workshop on Robust Subspace Learning and Applications in Computer Vision" at the 2017 "International Conference on Computer Vision".

The results in the paper use the following configuration flags:

myFlags = incAMFastPCPinputPars('TI_search');

myFlags.showFlag = 1;

myFlags.shrinkRule='l1proj';

myFlags.shrinkAlpha=0.75;

myFlags.ghost = 1;

myFlags.ghostUniModOff = 0.1;

myFlags.ghostFrames = 20;

myFlags.ghostFactor = 0;

myFlags.TI = 2; 

myFlags.TIextraOneLoopSolve = 1;

myFlags.folder_results = %folder to which to save the results

folder = % input folder

incrementalPCP_ball(folder,1, 3, 30, myFlags);

