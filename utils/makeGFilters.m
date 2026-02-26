function[F] = makeGfilters(cudaFlag)
%  
% Returns 4 Gaussian filters from the LML filter bank of size 49x49
%

if nargin < 1
  cudaFlag = 0;
end


  SUP=49;                 % Support of the largest filter (must be odd)
  SCALES=sqrt(2).^[1:4];

  if cudaFlag == 0
    F=zeros(SUP,SUP,length(SCALES));
  else
    F=zeros(SUP,SUP,length(SCALES), 'gpuArray' );
  end

  
  count=1;
  if cudaFlag == 0
    for i=1:length(SCALES),
      F(:,:,count)   = normalise(fspecial('gaussian',SUP,SCALES(i)));
      count=count+1;
    end
  else
    for i=1:length(SCALES),
      F(:,:,count)   = gpuArray( normalise(fspecial('gaussian',SUP,SCALES(i))) );
      count=count+1;
    end
  end
  
  
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function f=normalise(f), f=f-mean(f(:)); f=f/sum(abs(f(:))); return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

