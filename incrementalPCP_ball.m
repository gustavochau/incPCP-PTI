
function[D, L, S, stats] = incrementalPCP_ball(basedir, rank, innerLoops, winFrames, myFlags)
%  
%  [D, L] = incrementalPCP(basedir, iniK, rank, innerLoops)
%  
%  basedir      : directory where video frames are located
%  rank         : estimated rank (defaul 1)
%  innerLoops   : number of inner loops
%  winFrames    : number of frames 'to be remembered'
%  myFlags      : see incAMFastPCPinputPars.m file
%  
%  
%  %    binMaskMethod = INCPCP_COMPUTE_BINMASK_LS;

%  Examples: 
%  
%  % Standard PCP (fixed camera)
%  myFlags = incAMFastPCPinputPars('default');
%  [~, ~, ~, myStats] = incrementalPCP('./neovision3-1920x1088/', 1, 3, 50, myFlags);
%  
%  % Standard PCP (fixed camera) with CUDA-enabled functions
%  myFlags = incAMFastPCPinputPars('default'); myFlags.cudaFlag = 1;
%  [~, ~, ~, myStats] = incrementalPCP('./neovision3-1920x1088/', 1, 3, 50, myFlags);
%  
%  % Rigid transform invariant PCP (use myFlags.cudaFlag = 1 to use CUDA-enabled functions)
%  myFlags = incAMFastPCPinputPars('TI_search'); myFlags.baseTras = 10; myFlags.baseAlpha=4;
%  [~, ~, ~, myStats] = incrementalPCP('./lank3-640x480_T10-A05/', 1, 3, 50, myFlags);
%  
%  % Standard PCP + ghost suppression
%  myFlags = incAMFastPCPinputPars('default_ghost');
%  [~, ~, ~, myStats] = incrementalPCP('./neovision3-1920x1088/', 1, 2, 200, myFlags);
%  
%  
% Authors
% =======
% 
% Paul Rodriguez   prodrig@pucp.pe
% Brendt Wohlberg  brendt@lanl.gov
% 
%  
% Related papers
% ==============
%  
%  [1] Paul Rodriguez, Brendt Wohlberg, "A Matlab Implementation of a Fast Incremental 
%      Principal Component Pursuit Algorithm for Video Background Modeling", IEEE 
%      International Conference on Image Processing (ICIP), (Paris, France), October, 2014. 
%  
%  [2] Paul Rodriguez, Brendt Wohlberg, "Incremental Principal Component Pursuit for Video 
%      Background Modeling", Springer Journal of Mathematical Imaging and Vision 
%      (JMIV),  2015.
%  
%  [3] Paul Rodriguez, "Real-time Incremental Principal Component Pursuit for Video 
%      Background Modeling on the TK1", GPU Technical Conference (GTC), (San Jose, 
%      CA, USA), March, 2015.
%  
%  [4] Paul Rodriguez, Brendt Wohlberg, "Translational and Rotational Jitter Invariant 
%      Incremental Principal Component Pursuit for Video Background Modeling", IEEE 
%      International Conference on Image Processing (ICIP), (Quebec, Canada), September, 2015
%  
%  [5] Paul Rodriguez, Brendt Wohlberg, "Ghosting Suppression for Incremental Principal 
%      Component Pursuit Algorithms", submitted IEEE 
%      International Conference on Image Processing (ICIP), (Phoenix, AZ, USA), September, 2016
%  
%  
% Legal
% =====
% 
%  There is currently a patent pending that covers the incremental PCP method and
%  applications that is implemented in this source code.
%  
%  For non-commercial or academic use the source code of this program can be distributed 
%  and/or modified under the terms of the GNU Lesser General Public License (LGPL) version 3 
%  as published by the Free Software Foundation (http://opensource.org/licenses/lgpl-3.0.html).
%  
%  For other usage, please contact the authors.pcpPars
%  



% ======================
% Initial setup / checks
% ======================

addSubdirs('incrementalPCP_ball');

if nargin < 5
   myFlags = incAMFastPCPinputPars('default');
   if nargin < 4
      winFrames = 30;
      if nargin < 3
        innerLoops = 1;
      end
   end
end


%  --- Global flags ---
[grayFlag, showFlag, saveFlag, vecFlag, bgChangeFlag] = setGlobalFlags(myFlags);

lambda = myFlags.lambda;

ghostFrames = GS_setghostFrames(myFlags, winFrames);

folder_results=myFlags.folder_results;

if(myFlags.TI == 1)
    save_folder = 'vid_TI';
elseif(myFlags.TI == 2)
    save_folder = 'alnL_ball_gs';
else
    save_folder = 'vid_normal';
end

if (myFlags.grayFlag)
    save_folder=[save_folder '_gray']
else
    save_folder=[save_folder '_color']
end
    
    mkdir([folder_results '/' save_folder]);
% Get images / video properties
[Nrows Ncols nDims frames Imgs] = getImgsProperties(basedir, grayFlag, myFlags.url);

Ndata = Nrows*Ncols*nDims;
Npix  = Nrows*Ncols;


% URL case / default show images
[stats.kEnd, vmaxShow, vminShow] = set_Url_n_Show(myFlags, frames);




% ===============
% Initialization
% ===============

incAMFastPCPdefs;

% Generate filterBank for binary segmentation (superseded by Ghost-Suppression case)
[FBfreq, hr2FB, hc2FB, hDimsFB] = genFB(myFlags, Nrows, Ncols);



% --- CLK is ticking (init) ---
t = tic;
% -------------------------------    


if myFlags.fullInit 

  step = myFlags.stepInit;           % step (use in initialization)
  [U Sigma V] = initIncPCP(rank, step, winFrames, {Nrows, Ncols, nDims, frames, basedir, Imgs}, ...
                         {U, Sigma, V, rank}, showFlag, grayFlag, myFlags.url);

  stats.kIni = rank0+winFrames*step;
else % incremental intialization

  % read input video (myFlags.off_kIni --> start with an offset number of frames, default = 0)
  rank0 = 1;
  D = readBlockFrames(basedir, Imgs, [Nrows*Ncols*nDims, frames], 1+myFlags.off_kIni, rank0+myFlags.off_kIni, ...
                      grayFlag, vecFlag, myFlags.url, myFlags.cudaFlag);

  
%    [U Sigma] = qr(D(:), 0);  % D has rank0 columns (seems to have problems in Matlab R2015a)
  [U Sigma V] = svd(D(:), 0);  % D has rank0 columns
  Sigma = Sigma*V;
  
  if myFlags.cudaFlag == 0
    V = 1;
  else
    V = gpuArray(1);
  end 
  
  L = D;
  stats.kIni = rank0+1+myFlags.off_kIni;
end

stats.Tinit = toc(t);  % initialization elapsed time


% ---------------------------



% === CLK is ticking (global) ====
t = tic;
% ================================


for k=stats.kIni:stats.kEnd, 
  disp([num2str(k) '/' num2str(stats.kEnd)]);  

  t0 = tic;  % CLK per loop

  
  % read current frame
  curFrame = readBlockFrames(basedir, Imgs, [Nrows*Ncols*nDims, frames], k, k, grayFlag, vecFlag, myFlags.url, myFlags.cudaFlag);

  % if myFlags.TI == 0 --> alnFrame = curFrame; otherwise call TI (transform invariant)
  
  if(myFlags.TI == 2) %jacobian, modified by GC
      
      [alnL, my_h, my_hT, alphaEst] = get_TI_frame( L, curFrame, myFlags, Ncols, Nrows, nDims, Npix); 

      %% realign SVD to current frame
      for ww=1:size(V,1)
          ll = U*Sigma*(V(ww,:))';
          myll = reshape(ll, Nrows, Ncols, nDims);   
          myll=forwardTranform(myll, -alphaEst, my_hT, myFlags.alphaThresh, Nrows, Ncols, nDims, Npix, myFlags.cudaFlag);
          myll = (myll+curFrame)/2;
          if ww==1
              [U1 Sigma1] = qr(myll(:), 0);  % initialize L
              V1=1;
          else
              [U1 Sigma1 V1] = rank1IncSVD(U1, Sigma1, V1, myll(:), 1);  
          end
      end
      U=U1;
      V=V1;
      Sigma=Sigma1;
      [U Sigma V] = rank1IncSVD(U, Sigma, V, curFrame(:), 1);  

  else
      [alnFrame, my_h, my_hT, alphaEst] = get_TI_frame( curFrame, L, myFlags, Ncols, Nrows, nDims, Npix);
        [U Sigma V] = rank1IncSVD(U, Sigma, V, alnFrame(:), 1);  

  end

  %% -------------------------------------
  %% -------------------------------------


    % >>> Ghost case <<<
    if( myFlags.ghost )
    
      [vRowsLocal ~ ] = size(V);
      if( (k == stats.kIni) || ( vRowsLocal == 2 ))
      
        U2 = U; Sigma2 = Sigma; V2 = V;
              
      else 
      
        [U2 Sigma2 V2] = rank1IncSVD(U2, Sigma2, V2, curFrame(:), 1);
        
      end 
      
    end 
    % >>> ---------- <<<
    
    
    
  %% -------------------------------------
  %% -------------------------------------

  if(k > stats.kIni)
    Lold = L;
  else
    if myFlags.cudaFlag == 0
      Lold = [];
      Sold = [];      
      Sold2 = [];      
    else
      Lold = gpuArray([]);
      Sold = gpuArray([]);      
      Sold2 = gpuArray([]);      
    end
    Ldist = zeros(stats.kEnd - stats.kIni + 1, 1);
  end

  
  % ------------
  % inner loops
  % ------------
  S = L*0;
  for l=1:innerLoops,
  
  
    % compute current low-rank approximation ( if myFlags.TI == 0 --> L = myL = U S V')
    [L myL] = compute_LowRank(U, Sigma, V, alphaEst, my_h, Nrows, Ncols, nDims, Npix, myFlags);

    if( myFlags.ghost )
      [L2 myL2] = compute_LowRank(U2, Sigma2, V2, alphaEst, my_h, Nrows, Ncols, nDims, Npix, myFlags);
    end
    
    
    % compute current sparse approximation
    [S xx] = thresholding( curFrame - myL, lambda, myFlags.shrinkRule, myFlags.shrinkAlpha, myFlags.shrinkBeta, Sold);
    
    
    if( myFlags.ghost )
      [S2 xx2] = thresholding( curFrame - myL2, lambda, myFlags.shrinkRule, myFlags.shrinkAlpha, myFlags.shrinkBeta, Sold2);
    end 
    
    if (myFlags.TI ==2) % if we are on next loop
       [myL, my_h, my_hT, alphaEst] = get_TI_frame( myL, curFrame-S, myFlags, Ncols, Nrows, nDims, Npix); 
       if( myFlags.ghost )
           [myL2, my_h2, my_hT2, alphaEst] = get_TI_frame( myL, curFrame-S2, myFlags, Ncols, Nrows, nDims, Npix); 
       end
    end
    
    % Break condition
    if(l==innerLoops)  break; end

    
    if(l==1)
     
%       pFrame = alnFrame;  % 1st incSVD is applied to 'original' frame. NOTE: if myFlags.TI == 0 --> alnFrame = curFrame.
      pFrame = curFrame;  
      if( myFlags.ghost )
        pFrame2 = pFrame;
      end 
      
    else
    
      pFrame = r;
      if( myFlags.ghost )
        pFrame2 = r2;
      end 
      
    end
    
    
    % Compute residual. if myFlags.TI == 0 --> r = curFrame-S; 
    [r, myL, my_h, my_hT, alphaEst] = computeResidual(curFrame, S, L, myL, Nrows, Ncols, nDims, Npix, myFlags, ...
                                                      my_h, my_hT, alphaEst);
    if( myFlags.ghost )
      [r2, myL2, ~, ~, ~] = computeResidual(curFrame, S2, L2, myL2, Nrows, Ncols, nDims, Npix, myFlags, ...
                                                        my_h, my_hT, alphaEst);
    end
    
    if (myFlags.TI ==2) % if we are on next loop, realign r with base axis
        r = forwardTranform( r, -alphaEst, my_hT, myFlags.alphaThresh, Nrows, Ncols, nDims, Npix, myFlags.cudaFlag);
        if( myFlags.ghost )
            r2 = forwardTranform( r2, -alphaEst, my_hT, myFlags.alphaThresh, Nrows, Ncols, nDims, Npix, myFlags.cudaFlag);
        end
    end

    
    % rank-1 replace
    [U Sigma V] = rank1RepSVD(U, Sigma, V, rank, pFrame(:), r(:)); % check shkVideo idea    

    if( myFlags.ghost )
      [U2 Sigma2 V2] = rank1RepSVD(U2, Sigma2, V2, rank, pFrame2(:), r2(:));     
    end 
    
  end   % _END_ innerLoops
  % ----------------------

%       if rem(k-stats.kIni,2) == 0
%        Sold = S;
%       else
%        Sold = xx;
%       end

    Sold = xx;
    if( myFlags.ghost )
      Sold2 = xx2;
    end
    
  % ----------------------------
  % -- Compute masks (Ghost) ---
  % ----------------------------

  
  if( myFlags.ghost )

  
    [tmp] = binaryMaskGhost_1MOD(S2, [Nrows, Ncols, nDims], myFlags.ghostUniModOff, 1, myFlags.cudaFlag);
    fgMask2 = repmat(tmp, [1, nDims]);
    fgMask2 = reshape(fgMask2, [Nrows, Ncols, nDims]);
%    
    [tmp]  = binaryMaskGhost_1MOD(S, [Nrows, Ncols, nDims], myFlags.ghostUniModOff, 1, myFlags.cudaFlag);
    fgMask = repmat(tmp, [1, nDims]);
    fgMask = reshape(fgMask, [Nrows, Ncols, nDims]);
    
               
    % >>> Use masks to feedback and improve sparse estimate <<<


    bgMask = (1-fgMask) + (1-fgMask2) > 0;
    
    if saveFlag
      saveVideoFrame(bgMask, k, [folder_results '/' save_folder  '/bgmask_'], Nrows, Ncols, nDims, [], [], grayFlag, myFlags.cudaFlag);
    end
    
    lambdaM = lambda*(1-bgMask) + 2.0*lambda*bgMask;

    % NOTE: For the TI case, check that rM are aligned
    rM = r(:).*(1-bgMask(:)) + curFrame(:).*(bgMask(:));
    rM2 = r2(:).*(1-bgMask(:)) + curFrame(:).*(bgMask(:));

    if saveFlag
      saveVideoFrame(rM, k, [folder_results '/' save_folder  '/inTilde_'], Nrows, Ncols, nDims, [], [], grayFlag, myFlags.cudaFlag);
    end
%   

    rBG = 0.5*(rM + rM2);
    
    [vRowsLocal ~ ] = size(V);
    [U Sigma V] = rank1DwnSVD(U, Sigma, V, vRowsLocal);
    [U Sigma V] = rank1IncSVD(U, Sigma, V, rM(:), 1);  
    [L myL] = compute_LowRank(U, Sigma, V, alphaEst, my_h, Nrows, Ncols, nDims, Npix, myFlags);
%      S = shrink( curFrame(:) - myL(:), lambdaM(:));
    S = thresholding( curFrame(:) - myL(:), lambdaM(:), myFlags.shrinkRule, myFlags.shrinkAlpha, myFlags.shrinkBeta);

    [vRowsLocal ~ ] = size(V2);
    [U2 Sigma2 V2] = rank1DwnSVD(U2, Sigma2, V2, vRowsLocal);
    [U2 Sigma2 V2] = rank1IncSVD(U2, Sigma2, V2, rM(:), 1);  
    [L2 myL2] = compute_LowRank(U2, Sigma2, V2, alphaEst, my_h, Nrows, Ncols, nDims, Npix, myFlags);
%      S2 = shrink( curFrame(:) - myL2(:), lambdaM(:));
    S2 = thresholding( curFrame(:) - myL2(:), lambdaM(:), myFlags.shrinkRule, myFlags.shrinkAlpha, myFlags.shrinkBeta);
        
    
    % >>> ------------------------------------------------- <<<
  
  end 
  
  
  
  % ----------------------
  % -- Check BG change ---
  % ----------------------
    
  [Ldist U Sigma V bgChangeFlag vRows] = checkBGChange(k, stats.kIni, L, Lold, Ldist, U, Sigma, V, Nrows, Ncols, ...
                                                       bgChangeFlag, myFlags, curFrame);
    
  
  % ---------------
  % -- Downdate ---
  % ---------------

  if(vRows >= winFrames)    %downdate (1st col)
     [U Sigma V] = rank1DwnSVD(U, Sigma, V, 1);
  end

  if( myFlags.ghost && (vRows >= ghostFrames) )    %downdate (1st col)
     [U2 Sigma2 V2] = rank1DwnSVD(U2, Sigma2, V2, 1);
  end
  
  
  % ------------------------------------------------
  % -- Compute binary mask from sparse component ---
  % ------------------------------------------------
  
  if( myFlags.ghost )
    maskFB = 1 - bgMask;
  else
    maskFB = binaryMask(S, FBfreq, [Nrows, Ncols, nDims], [hr2FB, hc2FB, hDimsFB], k, stats, myFlags);
  end

  
  % -----   CLK per loop   -----  
  stats.Tframe{k} = toc(t0);
  % ----------------------------


  % ===================================
  %    Show / Save / Compute distances
  % ===================================


  % -- Apply inverse transform (i.e. align sparse approximation for display purposes)
  Sorig = S;
  if(myFlags.TI == 1)
     S = forwardTranform( S, -alphaEst, my_hT, myFlags.alphaThresh, Nrows, Ncols, nDims, Npix, myFlags.cudaFlag);
     if( myFlags.ghost )
      S2 = forwardTranform( S2, -alphaEst, my_hT, myFlags.alphaThresh, Nrows, Ncols, nDims, Npix, myFlags.cudaFlag);
     end
  end
  
  
  % compute distance
  if((k == stats.kIni) & (stats.kIni > 1) )    % set 'null' values for frames before kIni

    for mm = 1:k-1
      stats.dist{mm} = 0;
    end
    
    stats.Fmeasure(1:k-1) = -1; stats.recall(1:k-1) = -1; stats.precision(1:k-1) = -1;
    stats.TP(1:k-1) = -1; stats.FP(1:k-1) = -1; stats.FN(1:k-1) = -1;
    
  else
    [stats.dist{k}, stats.Fmeasure(k), stats.recall(k), stats.precision(k) ...
                  stats.TP(k), stats.FP(k), stats.FN(k) ] = computeDistance(S, maskFB, myFlags, k, ...
                                                                          Nrows, Ncols, nDims);
    
%      [k stats.dist{k}]
    
    if( isnan( stats.dist{k} ) )
      stats.dist{k} = stats.dist{k-1};
    end 
    
  end
  
  
  % compute maximum/minimum (sparse approximation, for display purposes)
  if myFlags.adaptShow
     vmaxShow = max([S(:); vmaxShow]);
     vminShow = min([S(:); vminShow]);
  end    

  
  % save data
  if saveFlag
    saveVideoFrame(S, k, [folder_results '/' save_folder '/S_incPCP_'], Nrows, Ncols, nDims, vminShow, vmaxShow, grayFlag, myFlags.cudaFlag);
    saveVideoFrame(L, k, [folder_results '/' save_folder '/L_incPCP_'], Nrows, Ncols, nDims, [], [], grayFlag, myFlags.cudaFlag);

    if( myFlags.ghost )
      saveVideoFrame(S2, k, [folder_results '/' save_folder '/S2_incPCP_'], Nrows, Ncols, nDims, vminShow, vmaxShow, grayFlag, myFlags.cudaFlag);
      saveVideoFrame(L2, k, [folder_results '/' save_folder '/L2_incPCP_'], Nrows, Ncols, nDims, [], [], grayFlag, myFlags.cudaFlag);
    end 
    
    saveFrame2Mat(Sorig, k, [folder_results '/' save_folder '/S_incPCP_'], myFlags.cudaFlag);
    if(myFlags.ComputeBinaryMask > 0)
      saveFrame2Mat(maskFB, k, [folder_results '/' save_folder '/Smask_incPCP_'], myFlags.cudaFlag);
    end
    
    if myFlags.url == 1
      saveVideoFrame(curFrame, k, './vid/Orig_', Nrows, Ncols, nDims, [], [], grayFlag, myFlags.cudaFlag);    
    end
  end

  
  % Show current frames
  if showFlag

    figure(4); imagesc( reshape( showNormalize(S, vminShow, vmaxShow, grayFlag), [Nrows, Ncols, nDims])); colormap gray;
    
    figure(5); imagesc( reshape( showNormalize(curFrame), [Nrows, Ncols, nDims])); colormap gray;

    
    
    if(myFlags.ComputeBinaryMask > 0)
      figure(6); imagesc( maskFB ); colormap gray;
    end 
    
    drawnow;
  end


end     % _END_ FOR(k)

  
% -----   CLK (global)   -----  
stats.Tfull = toc(t);
% ----------------------------


return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[binMask] = binaryMask_1MOD(S, imgProps, myFlags, mmFlag)

if nargin < 4
  mmFlag = 0;
end

  [tmpFB, tau] = unimodal(abs(S), myFlags.uniModOff, myFlags.cudaFlag);
  
  if( myFlags.vecFlag == 1)
    tmpFB = reshape(tmpFB, imgProps);
  end
  
if mmFlag == 1

  
  SE = [0 1 0; 1 1 1; 0 1 0];
  
  if( imgProps(3) == 3 )
  
    tmpFB(:,:,1) = imerode( tmpFB(:,:,1), SE );
    tmpFB(:,:,2) = imerode( tmpFB(:,:,2), SE );
    tmpFB(:,:,3) = imerode( tmpFB(:,:,3), SE );
  
    tmpFB(:,:,1) = imdilate( tmpFB(:,:,1), SE );
    tmpFB(:,:,2) = imdilate( tmpFB(:,:,2), SE );
    tmpFB(:,:,3) = imdilate( tmpFB(:,:,3), SE );
  
    binMask = sum(tmpFB, 3) > 0;
  
  else
  
    tmpFB = imerode( tmpFB, SE );  
    tmpFB = imdilate( tmpFB, SE );
  
    binMask = sum(tmpFB, 3) > 0;
    
  end 
  
  binMask = imerode( binMask, SE );
  binMask = imdilate( binMask, SE );
  
else

  binMask = sum(tmpFB, 3) > 0;

end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[binMask] = binaryMaskGhost_1MOD(S, imgProps, uniModOff, mmFlag, cudaFlag)

if nargin < 4
  mmFlag = 0;
end

  [tmpFB, tau] = unimodal(abs(S), uniModOff, cudaFlag);
  tmpFB = reshape(tmpFB, imgProps);
  
if mmFlag == 1

  
  SE = [0 1 0; 1 1 1; 0 1 0];
  
  if( imgProps(3) == 3 )
  
    tmpFB(:,:,1) = imerode( tmpFB(:,:,1), SE );
    tmpFB(:,:,2) = imerode( tmpFB(:,:,2), SE );
    tmpFB(:,:,3) = imerode( tmpFB(:,:,3), SE );
  
    tmpFB(:,:,1) = imdilate( tmpFB(:,:,1), SE );
    tmpFB(:,:,2) = imdilate( tmpFB(:,:,2), SE );
    tmpFB(:,:,3) = imdilate( tmpFB(:,:,3), SE );
  
    tmpFB(:,:,1) = imdilate( tmpFB(:,:,1), SE );
    tmpFB(:,:,2) = imdilate( tmpFB(:,:,2), SE );
    tmpFB(:,:,3) = imdilate( tmpFB(:,:,3), SE );
    
    binMask = sum(tmpFB, 3) > 0;
  
  else
  
    tmpFB = imerode( tmpFB, SE );  
    tmpFB = imdilate( tmpFB, SE );
    tmpFB = imdilate( tmpFB, SE );
  
    binMask = sum(tmpFB, 3) > 0;
    
  end 
  
  binMask = imerode( binMask, SE );
  binMask = imdilate( binMask, SE );
  binMask = imdilate( binMask, SE );
  binMask = imdilate( binMask, SE );
  binMask = imdilate( binMask, SE );
  binMask = imerode( binMask, SE );
  binMask = imerode( binMask, SE );
  binMask = imerode( binMask, SE );
  
else

  binMask = sum(tmpFB, 3) > 0;

  
end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function[binMask] = binaryMask_LM(S, FBfreq, imgProps, kernelProps, k, stats, myFlags)

    Nrows = imgProps(1);
    Ncols = imgProps(2);
    nDims = imgProps(3);
  
    hr2FB   = kernelProps(1);
    hc2FB   = kernelProps(2);
    hDimsFB = kernelProps(3);
  

%      Stmp = S(1:2:end, 1:2:end, :);
%      [Nr2 Nc2 nD2 ] = size(Stmp);
    
    if(k == stats.kIni)
      if( myFlags.cudaFlag == 0)
        mFB = zeros(Nrows*Ncols, hDimsFB*nDims);    
%          mFB = zeros(Nr2*Nc2, hDimsFB*nDims);    
      else
        mFB = zeros(Nrows*Ncols, hDimsFB*nDims, 'gpuArray');    
%          mFB = zeros(Nr2*Nc2, hDimsFB*nDims, 'gpuArray');    
      end
    end

    if( myFlags.vecFlag == 1)
      Fsp = fft2(reshape(S, imgProps), Nrows+hr2FB, Ncols+hc2FB);    
    else
      Fsp = fft2(S, Nrows+hr2FB, Ncols+hc2FB);
    end
    
%      Fsp = fft2(Stmp, Nr2+hr2FB, Nc2+hc2FB);
    
    if( nDims == 3 )

      % apply FBank
      U1 = bsxfun(@times, Fsp(:,:,1), FBfreq);
      U2 = bsxfun(@times, Fsp(:,:,2), FBfreq);
      U3 = bsxfun(@times, Fsp(:,:,3), FBfreq);
  
      % compute IFFT
      u1 = ifft2(U1, 'symmetric');
      u2 = ifft2(U2, 'symmetric');
      u3 = ifft2(U3, 'symmetric');
    
    
      % group data
      for l=1:hDimsFB,

        z = u1(hr2FB:hr2FB+Nrows-1,hc2FB:hc2FB+Ncols-1,l); 
%          z = u1(hr2FB:hr2FB+Nr2-1,hc2FB:hc2FB+Nc2-1,l); 
        mFB(:,3*(l-1)+1) = z(:);

        z = u2(hr2FB:hr2FB+Nrows-1,hc2FB:hc2FB+Ncols-1,l); 
%          z = u2(hr2FB:hr2FB+Nr2-1,hc2FB:hc2FB+Nc2-1,l); 
        mFB(:,3*(l-1)+2) = z(:);

        z = u3(hr2FB:hr2FB+Nrows-1,hc2FB:hc2FB+Ncols-1,l); 
%          z = u3(hr2FB:hr2FB+Nr2-1,hc2FB:hc2FB+Nc2-1,l); 
        mFB(:,3*(l-1)+3) = z(:);
    
      end   
    
    
    else % _ELSE_ IF(nDims)
    
    
      % apply FBank
      U1 = bsxfun(@times, Fsp, FBfreq);
  
      % compute IFFT
      u1 = ifft2(U1, 'symmetric');
    
      % group data      
      for l=1:hDimsFB,

        z = u1(hr2FB:hr2FB+Nrows-1,hc2FB:hc2FB+Ncols-1,l); 
%          z = u1(hr2FB:hr2FB+Nr2-1,hc2FB:hc2FB+Nc2-1,l); 
        mFB(:,l) = z(:);

      end 
      
      
    end % -------------------------------------
    
    tmpFB  = reshape( var(mFB,[], 2), [Nrows, Ncols] );
%      tmpFB  = reshape( var(mFB,[], 2), [Nr2, Nc2] );

    [binMask, tau] = unimodal(tmpFB, 0.025, myFlags.cudaFlag);
    
%      if( myFlags.cudaFlag == 0)
%        maskFB = imresize(maskFB2, [Nrows Ncols], 'nearest');
%      else
%        maskFB = imresize(gather(maskFB2), [Nrows Ncols], 'nearest');
%      end 
    
%      % FIXME: add call for unimodal segmentation
%      if( myFlags.cudaFlag == 0)
%        maskFB = reshape( (tmpFB > myFlags.binaryThresh), [Nrows, Ncols] );
%      else
%        maskFB = gather( reshape( (tmpFB > myFlags.binaryThresh), [Nrows, Ncols] ) );
%      end
    

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[binMask] = binaryMask(S, FBfreq, imgProps, kernelProps, k, stats, myFlags)

incAMFastPCPdefs;

  switch(myFlags.ComputeBinaryMask)
  
    case{INCPCP_COMPUTE_BINMASK_NONE}
      binMask = nan;
      
    case{INCPCP_COMPUTE_BINMASK_LM}
      binMask = binaryMask_LM(S, FBfreq, imgProps, kernelProps, k, stats, myFlags);

    case{INCPCP_COMPUTE_BINMASK_1MOD}
      binMask = binaryMask_1MOD(S, imgProps, myFlags);

    otherwise
      disp('Should not happen (binaryMask, undefined method)');
      binMask = nan;
  end

  
  

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[localDist U Sigma V bgChangeFlag vRows] = checkBGChange(k, kIni, L, Lold, localDist, U, Sigma, V, Nrows, Ncols, bgChangeFlag, myFlags, curFrame)


  [vRows ~ ] = size(V);

  
  % -- check for changes in background
  if(k > kIni)
    if myFlags.cudaFlag == 0
      localDist(k) = sum(abs(L(:) - Lold(:))) / (Nrows*Ncols);
    else
      localDist(k) = gather(sum(abs(L(:) - Lold(:)))) / (Nrows*Ncols);      
    end
  end


  if( (k > kIni+1) )
    Lfrac = localDist(k) / localDist(k-1);
  else
    Lfrac = 1;
  end

  if( myFlags.TI == 1)
    Lfrac = Lfrac*myFlags.TIadjustDiff;  % scale down this fraction since for TI case, diferences could be due misalignement
  end

  % verbose
%    [Lfrac vRows]
  
  
  % If fraction Lfrac is greater than threshold and
  % (i) background is considered stable then re-initialize or 
  % (ii) a change has recently been detectect (this could mean that bg is changing so we 
  %      do not have to  wait until we considere it stable) then re-initialize
      
  if( (Lfrac > myFlags.backgroundThresh) & ( (vRows >= myFlags.backgroundStable) | bgChangeFlag ) )  

    [U Sigma] = qr(curFrame(:), 0);  % D has rank0 columns

    if myFlags.cudaFlag == 0
      V = 1;
    else
      V = gpuArray(1);
    end
    
    vRows = 1;
    
    bgChangeFlag = 1;

  end  
  
  if vRows >= myFlags.backgroundStable
    bgChangeFlag = 0;
  end
  
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[r, myL, my_h, my_hT, alphaEst] = computeResidual(curFrame, S, L, myL, Nrows, Ncols, nDims, Npix, myFlags, my_h, my_hT, alphaEst)

    if(myFlags.TI == 1)
    
      if(myFlags.TIextraOneLoopSolve == 1)

      incAMFastPCPdefs;
      
        switch myFlags.TIextraOneLoopStrategy
        
          case{INCPCP_TI_REFINEMENT_STANDARD}
            [~, my_h, my_hT, alphaEst] = oneLoopTI_solve(curFrame-S, L, myFlags, Nrows, Ncols, nDims, Npix);
      
          case{INCPCP_TI_REFINEMENT_SEARCH}
            alphaEst = angle_refinement(curFrame, S, L, Nrows, Ncols, nDims, Npix, myFlags, my_h, alphaEst);
                      
        end % _END_ SWITCH
        
        myL = forwardTranform(L, alphaEst, my_h, myFlags.alphaThresh, Nrows, Ncols, nDims, Npix, myFlags.cudaFlag);
            
      end % _END_ IF( TIextraOneLoopSolve )
      
                                        
      r = oneLoop_IHT(curFrame, myL, S, alphaEst, myFlags.alphaThresh, my_hT, Nrows, Ncols, nDims, Npix, myFlags.cudaFlag);      
      r = L - r;
      
      
    else  %---------------------------------------

      r = curFrame-S;
            
    end  %---------------------------------------

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[alphaEst] = angle_refinement(curFrame, S, L, Nrows, Ncols, nDims, Npix, myFlags, my_h, alphaEst)

        
  pres = myFlags.TIfibonacciThresh;

  if abs(alphaEst) > 0
     alphaI = pres*floor( alphaEst / pres);
     alphaE = pres*ceil( alphaEst / pres);
  else
     alphaI = -pres/2;
     alphaE =  pres/2;
  end
        
  range = alphaE - alphaI;
  alpha = alphaI:range/(myFlags.TIAngleSearchBins-1):alphaE;
     
  if myFlags.cudaFlag == 0
    lDist = zeros(myFlags.TIAngleSearchBins,1);
  else
    lDist = zeros(myFlags.TIAngleSearchBins,1, 'gpuArray');
  end
     
     
  for k=1:myFlags.TIAngleSearchBins

      tmp   = convRotXYfull_img(L, alpha(k), 1, myFlags.cudaFlag);
      cFest = myConv3(tmp, my_h, 'same', Nrows, Ncols, nDims, Npix, myFlags.cudaFlag ) + S;

%        lDist(k) = norm( cFest(:) - curFrame(:) );
      lDist(k) = sum( ( cFest(:) - curFrame(:) ).^2 );
  end
        
  [dummy pos] = min(lDist);
  alphaEst = alpha(pos);

  
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[L myL] = compute_LowRank(U, Sigma, V, alphaEst, my_h, Nrows, Ncols, nDims, Npix, myFlags) 
    
    if(myFlags.TI == 1)
      L = reshape( U*Sigma*(V(end,:)'), Nrows, Ncols, nDims);
      myL = forwardTranform(L, alphaEst, my_h, myFlags.alphaThresh, Nrows, Ncols, nDims, Npix, myFlags.cudaFlag);

    else
      L = U*Sigma*(V(end,:)');
      
      if myFlags.vecFlag == 0
         L = reshape(L, Nrows, Ncols, nDims);         
      end 
      
      myL = L;
      
    end
    
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[alnFrame, my_h, my_hT, alphaEst] = get_TI_frame( curFrame, L, myFlags, Ncols, Nrows, nDims, Npix)

  if(myFlags.TI >= 1)
  
    [alnFrame, my_h, my_hT, alphaEst] = oneLoopTI_solve(curFrame, L, myFlags, Nrows, Ncols, nDims, Npix);
  
  else

    alnFrame = curFrame;

    % not used in this case
    my_h = nan;
    my_hT = nan;
    alphaEst = nan;
    
  end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[dist, fmeasure, recall, precision, TP, FP, FN] = computeDistance(S, maskFB, myFlags, k, Nrows, Ncols, nDims, verbose)

if nargin < 8
  verbose = 0;
end

incAMFastPCPdefs;


  switch(myFlags.sparseGTflag)
  
    case{INCPCP_GT_NONE}
      % do nothing
      dist = nan;
      fmeasure = nan;
      recall = nan;
      precision = nan;
      TP = nan;
      FP = nan;
      FN = nan;
    
      % ---------------------
      % ---------------------
      
      
    case{INCPCP_GT_MATFILES}
      dist = computeDistanceMatFiles(S, myFlags, k, Nrows, Ncols, nDims, verbose);

      fmeasure = nan;
      recall = nan;
      precision = nan;
      TP = nan;
      FP = nan;
      FN = nan;
      
      % ---------------------
      % ---------------------
      
      
    case{INCPCP_GT_SPARSEMAT}
      
      disp('INCPCP_GT_SPARSEMAT not yet implemented');
      
      dist = nan;

      fmeasure = nan;
      recall = nan;
      precision = nan;
      TP = nan;
      FP = nan;
      FN = nan;

      % ---------------------
      % ---------------------
      
      
    case{INCPCP_GT_BINMASK}
      [fmeasure, recall, precision, TP, FP, FN] = computeDistanceFmeasure(S, maskFB, myFlags, k, Nrows, Ncols, nDims, verbose);
      dist = nan;
    
      % ---------------------
      % ---------------------
      
    case{INCPCP_GT_BINMASK_MATFILES}
      
      [fmeasure, recall, precision, TP, FP, FN] = computeDistanceFmeasure(S, maskFB, myFlags, k, Nrows, Ncols, nDims, verbose);

      dist = computeDistanceMatFiles(S, myFlags, k, Nrows, Ncols, nDims, verbose);
      
      % ---------------------
      % ---------------------
      
    otherwise
      disp('GT format is not recognize... ');
      
    
  end % _END_ SWITCH

  
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[dist] = computeDistanceMatFiles(S, myFlags, k, Nrows, Ncols, nDims, verbose)

if nargin < 7
  verbose = 0;
end


    if( isdir( myFlags.GTBasedirMat ) )   % 

       GT = getMatFile(myFlags.GTBasedirMat, k);

       if( length(GT) > 1 )
       
        GT = reshape(GT, Nrows, Ncols, length(GT)/(Nrows*Ncols));

        if(myFlags.grayFlag==1) 
            GT = mean(GT,3); 
        end

        if( ~isempty(GT) )
          dist = sum( abs( S(:) - GT(:) ) ) / (Nrows*Ncols);
          if verbose==1
             [k dist]
          end
        else
          disp('computeDistance: GT is empty ... should not happend');
          dist = nan;
        end
        
      else % _ELSE_ IF(length(GT) > 1)
        disp('computeDistance: GT is size 1 ... should not happend');
        dist = nan;
      end 
      
    else
%        myFlags.GTBasedirMat
      disp('computeDistance: GT is not a dir ... should not happend');
      dist = nan;
          
    end  % _END_ IF ~iscell( S_GT )


return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[fmeasure, recall, precision, TPk, FPk, FNk] = computeDistanceFmeasure(S, maskFB, myFlags, k, Nrows, Ncols, nDims, verbose)

if nargin < 8
  verbose = 0;
end

    if( (k >= myFlags.binaryfIndex(1,2)) & (k <= myFlags.binaryfIndex(end,2)))

      [~, ~, ~, framesGTbin, ImgsGTbin] = getImgsProperties(myFlags.GTBasedir);
    
    
      m = find( myFlags.binaryfIndex(:,2) == k );
      
      if( length(m) == 1 )
      
        GTbin = readBlockFrames(myFlags.GTBasedir, ImgsGTbin, [Nrows*Ncols, 0], ...
                              myFlags.binaryfIndex(m,1), myFlags.binaryfIndex(m,1), ...
                              0, 0, 0, 0);

        [gtNr, gtNc, gtDims] = size(GTbin);
        if( gtDims == 3)
          GTbin = rgb2gray(GTbin);
        end
      
%  %          % >>> debgug <<<
%  %          [k m myFlags.binaryfIndex(m,1)]
%  %          figure(myFlags.binaryShowFlag); imagesc(GTbin); colormap gray;
%  %          size(GTbin)
%  %          size(maskFB)
%  %          % >>> ------ <<<

        [~, ~, mask_nDims] = size( maskFB );
        if mask_nDims == 1
          [fmeasure, precision, recall, TPk, FPk, FNk] = f_measure(GTbin>0, maskFB);

          if(myFlags.binaryShowFlag > 0)
            figure(myFlags.binaryShowFlag); imagesc( 0.5*( (GTbin>0) + maskFB ) ); colormap gray;  %imshowpair(GTbin>0, maskFB);    
          end 
          
        else
          [fmeasure, precision, recall, TPk, FPk, FNk] = f_measure(GTbin>0, sum(maskFB,3)>0 );

          if(myFlags.binaryShowFlag > 0)
            figure(myFlags.binaryShowFlag); imagesc( 0.5*( (GTbin>0) + sum(maskFB,3)>0 ) ); colormap gray;  %imshowpair(GTbin>0, maskFB);    
          end 
        
        
        end

      else % _ELSE_ IF( length(m) == 1 )
         
        fmeasure = -1;
        precision = -1;
        recall = -1;
        TPk = -1;
        FPk = -1;
        FNk = -1;
      end

      if verbose
      
        % FIXME: check this to show cumulative F-m
%          TP = TP + TPk;
%          FP = FP + FPk;
%          FN = FN + FNk;
%        
%          % Precision
%          Pk = TP/(TP+FN);
%  
%          % Recall
%          Rk = TP/(TP+FP);
%  
%          % F-measure
%          Fk = (2*Pk*Rk)/(Pk+Rk);
%        
%          [k Fk Pk TP FN FP]
        
      end % _END_ verbose 
      
      
    else
    
      fmeasure = -1;
      precision = -1;
      recall = -1;
      TPk = -1;
      FPk = -1;
      FNk = -1;
      
    end % _END_ IF(k)

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[alnFrame, my_h, my_hT, alphaEst] = oneLoopTI_solve(shkFrame, L, myFlags, Nrows, Ncols, nDims, Npix)

    hDim = myFlags.baseTras;
    alphaBase = myFlags.baseAlpha;

    % solve current endogenous sparse representation
    [my_h my_hT] = csr4incPCP(L, shkFrame, myFlags.TImu, myFlags.TIcsrLoops, hDim, myFlags.TIcsrThresh, 0, 0, myFlags.cudaFlag);

    % --- Apply "inverse" translation ---
    tmp = myConv3( shkFrame(:), my_hT, 'same', Nrows, Ncols, nDims, Npix, myFlags.cudaFlag );


    % solve current rot via conv transformation (fibonacci)
    % Find alphaEst s.t. Rot(L, alphaEst) = tmp;
     myfun = @(gamma) costRotXYfull(gamma, L, tmp, 1, myFlags.cudaFlag);
     alphaEst = FSearch(myfun, -alphaBase, alphaBase, myFlags.TIfibonacciThresh);

     
    % Improve current Transformation

     if abs(alphaEst) > myFlags.alphaThresh

        % --- Apply forward rotation
        tmp = convRotXYfull_img(L, alphaEst, 1, myFlags.cudaFlag);
      
        % --- Solve (again) for translation ---
        [my_h my_hT] = csr4incPCP(tmp, shkFrame, myFlags.TImu, myFlags.TIcsrLoops, hDim, myFlags.TIcsrThresh, 0, 0, myFlags.cudaFlag);

        % --- Apply "inverse" translation ---
        tmp = myConv3( shkFrame(:), my_hT, 'same', Nrows, Ncols, nDims, Npix, myFlags.cudaFlag);


        myfun = @(gamma) costRotXYfull(gamma, L, tmp, 1, myFlags.cudaFlag);
        alphaEst = FSearch(myfun, -alphaBase, alphaBase, myFlags.TIfibonacciThresh);

        alnFrame = convRotXYfull_img(tmp, -alphaEst, 1, myFlags.cudaFlag);

     else   
      alphaEst = 0;
      alnFrame = tmp;
     end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[Lshk] = forwardTranform(L, alphaEst, my_h, alphaThresh, Nrows, Ncols, nDims, Npix, cudaFlag)

    if abs(alphaEst) > alphaThresh

      % apply rigid transformation B = T(R(U))
      Lshk = convRotXYfull_img( L, alphaEst, 1, cudaFlag);    
      Lshk = myConv3(Lshk(:), my_h, 'same', Nrows, Ncols, nDims, Npix, cudaFlag);

    else
      Lshk = myConv3(L(:), my_h, 'same', Nrows, Ncols, nDims, Npix, cudaFlag);
    end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[r] = oneLoop_IHT(shkFrame, Lshk, S, alphaEst, alphaThresh, my_hT, Nrows, Ncols, nDims, Npix, cudaFlag)

      

      r = Lshk - (shkFrame - S);   % A*X - B

      if abs(alphaEst) > alphaThresh
        r = myConv3(r(:), my_hT, 'same', Nrows, Ncols, nDims, Npix, cudaFlag);     % translation
        r = convRotXYfull_img( r, -alphaEst, 1, cudaFlag);
      else
        r = myConv3(r(:), my_hT, 'same', Nrows, Ncols, nDims, Npix, cudaFlag);     % translation
      end 

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[dist] = GTisSparseApproximation(myFlags, S, Nrows, Ncols, k, oldist)


[ ismatrix( myFlags.sparseGT )  isnumeric( myFlags.sparseGT ) ]

    % ----
    if( ismatrix( myFlags.sparseGT ) & isnumeric( myFlags.sparseGT ) )
      dist = sum( abs( S - myFlags.sparseGT(:,k) ) ) / (Nrows*Ncols);

    else

      % ----
      if isdir ( myFlags.sparseGT )

      1

        GT = getMatFile(myFlags.sparseGT, k);
        if(length(GT) > 1)
          if( ~isempty(GT) )
            dist = sum( abs( S - GT ) ) / (Nrows*Ncols);
          end
        else
            dist = olddist;
        end

      end % _END_ isdir

    end % _END_ if( ismatrix( S_GT ) & isnumeric( S_GT ) )

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[Fm, Recall, Precision] =  GTisManual(myFlags, S, Nrows, Ncols, k)

  myFlags.sparseGT;
  framesGT = myFlags.sparseGT{1}; 
  fnamesGT = myFlags.sparseGT{2};

  Fm = 0;
  Recall = 0;
  Precision = 0;

  if( sum( (framesGT - k - myFlags.frameNoff) == 0 ) == 1 )

    n = find( (framesGT - k - myFlags.frameNoff) == 0 );
    fname  = fnamesGT(n).name;
    
  end

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function[U Sigma V] = initIncPCP(k0, step, localNFrames, imgProps, cSVD, showFlag, grayFlag, urlFlag, saveFlag)

if nargin < 9
  saveFlag = 0;
  if nargin < 8
    urlFlag = 0;
    if nargin < 7
      grayFlag = 0;
    end
  end
end

Nrows   = imgProps{1};
Ncols   = imgProps{2};
nDims   = imgProps{3};
frames  = imgProps{4};
basedir = imgProps{5};
Imgs    = imgProps{6};

U     = cSVD{1};
Sigma = cSVD{2};
V     = cSVD{3};
rank  = cSVD{4};

rank0 = length(Sigma);

for k=k0:step:k0+localNFrames*step, %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  if(k>frames) break; end;

  curFrame = readBlockFrames(basedir, Imgs, [Nrows*Ncols*nDims, frames], k, k, grayFlag, urlFlag);

  if rank0 < rank
    [U Sigma V ] = rank1IncSVD(U, Sigma, V, curFrame, 0);   % rank increasing
    rank0 = rank0+1;
  else
    [U Sigma V ] = rank1IncSVD(U, Sigma, V, curFrame, 1); 
  end

  if showFlag
    L = U*Sigma*(V(end,:)');
    figure(1); imagesc( reshape( Normalize(L), [Nrows, Ncols, nDims])); colormap gray;

    if saveFlag
      saveVideoFrame(L, k, './vid/L_incPCP_', Nrows, Ncols, nDims);
    end

  end



end

% generate rank 'average' backgrounds
V = mean(V,1);
V = repmat(V, rank, 1);incAMFastPCPdefs;


return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function[] = addSubdirs(fname)
% Add all subdirectories of the parent directory of this
% script into the path

p0 = which(fname);
K = strfind(p0, filesep);
p1 = p0(1:K(end)-1);

mypath = genpath(p1);
path(path,mypath);


%  clear p0 K mypath p1
clear p0 K mypath p1 

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[ghostFrames] = GS_setghostFrames(myFlags, winFrames)
% Set parameters if 'ghost' is enabled
if myFlags.ghost == 1
  if (myFlags.ghostFactor == 0)
    ghostFrames = myFlags.ghostFrames;
  else
    ghostFrames = winFrames / myFlags.ghostFactor;
  end
else
  ghostFrames = 0;
end 

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[grayFlag, showFlag, saveFlag, vecFlag, bgChangeFlag] = setGlobalFlags(myFlags)

grayFlag = myFlags.grayFlag;
showFlag = myFlags.showFlag;
saveFlag = myFlags.saveFlag;

bgChangeFlag = 0;


if myFlags.cudaFlag == 1
  myFlags.vecFlag = 0;
end

vecFlag = myFlags.vecFlag;

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[kEnd, vmaxShow, vminShow] = set_Url_n_Show(myFlags, frames)
if myFlags.url
  if frames == -1
    kEnd = 10000;
  else    
    kEnd = frames;
  end

else
    kEnd = frames;
end


if myFlags.adaptShow
  if myFlags.cudaFlag == 0
    vmaxShow = -1e10;
    vminShow =  1e10;
  else
    vmaxShow = gpuArray(-1e10);
    vminShow = gpuArray(1e10);
  end
end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[FBfreq, hr2FB, hc2FB, hDimsFB] = genFB(myFlags, Nrows, Ncols)
%  

incAMFastPCPdefs;

if(myFlags.ComputeBinaryMask == INCPCP_COMPUTE_BINMASK_LM)

  FB = makeLMfilters(myFlags.cudaFlag);
  [hrFB, hcFB, hDimsFB] = size(FB);
  
  hr2FB = floor(hrFB/2);
  hc2FB = floor(hcFB/2);

  FBfreq = fbank2freq(FB, Nrows, Ncols, myFlags.cudaFlag);        
  
else
  
  FBfreq = [];
  hr2FB = []; hc2FB = []; hDimsFB = [];
  
end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[x x1] = thresholding(z, lambda, shrinkRule, alpha, beta, Sold );

if nargin < 6
  Sold = [];
  if nargin < 5
    beta = 0.0;
    if nargin < 5
      alpha = 1.0;
      if nargin < 5
        shrinkRule = 'soft';
      end 
    end
  end 
end

x1 = [];

switch(shrinkRule)
  
  %  case{'nng'}
  %    x = nngth( z + Mu*v, Mu*lambda);
  %  
  %  case{'hard'}
  %    x = hardth( z + Mu*v, Mu*lambda);

  case{'soft'}
    x = shrink(z, lambda);

  case{'softPnng'}
    x = shrinkNng(z, lambda, alpha, beta);

  case{'l1projSTD'}

    gamma = sum(abs(z(:)));
    z = z.*(abs(z) > 0.01);
    
    [x th] = projL1(z(:), 0.9*gamma, 1e-8, 20);

    
    
  case{'l1proj'}
  
    [Nr, Nc, nDims] = size(z);

    z = z.*(abs(z) > 0.01);
    gamma = sum(abs(z(:)));
    [x1 th] = projL1(z(:), alpha*gamma, 1e-8, 20);
    
    x1 = reshape(x1, size(z));
    
    
    if ~isempty(Sold)
      yy = x1 - Sold;

      if nDims == 3
        [masky, tau] = unimodal(sum(abs(yy),3), 0, 0);

        masky = imerode(masky, [0 1 0; 1 1 1; 0 1 0]);
        masky = imdilate(masky, [0 1 0; 1 1 1; 0 1 0]);
        masky = imdilate(masky, [0 1 0; 1 1 1; 0 1 0]);
      
        masky = 1.0 - masky;
      
        bb(:,:,1) = x1(:,:,1).*masky;
        bb(:,:,2) = x1(:,:,2).*masky;
        bb(:,:,3) = x1(:,:,3).*masky;
      
        gamma = sum(abs(bb(:)));
      
        [yy th2] = projL1(bb(:), 0.3*gamma, 1e-8, 20, 0);

        yy = reshape(yy, size(z));
       
        x(:,:,1) = x1(:,:,1).*(1-masky) + yy(:,:,1);
        x(:,:,2) = x1(:,:,2).*(1-masky) + yy(:,:,2);
        x(:,:,3) = x1(:,:,3).*(1-masky) + yy(:,:,3);
        
      else
      
        [masky, tau] = unimodal(abs(yy), 0, 0);

        masky = imerode(masky, [0 1 0; 1 1 1; 0 1 0]);
        masky = imdilate(masky, [0 1 0; 1 1 1; 0 1 0]);
        masky = imdilate(masky, [0 1 0; 1 1 1; 0 1 0]);
      
        masky = 1.0 - masky;
      
        bb = x1.*masky;
      
        gamma = sum(abs(bb(:)));
      
        [yy th2] = projL1(bb(:), 0.3*gamma, 1e-8, 20, 0);

        yy = reshape(yy, size(z));
       
        x = x1.*(1-masky) + yy;
        
      end 
      

    else
      x = x1;
    end
%      
     
    
  otherwise
    disp('threshFlag: Unknown method.')
    x = [];
    
end


return;

