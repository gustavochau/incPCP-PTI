function[F] = makeLMfilters(cudaFlag, SUP)
%  
% Returns the LML filter bank of size 49x49x48 in F. To convolve an
% image I with the filter bank you can either use the matlab function
% conv2, i.e. responses(:,:,i)=conv2(I,F(:,:,i),'valid'), or use the
% Fourier transform.
%  
if nargin < 2
    SUP = 49;
  if nargin < 1
    cudaFlag = 0;
  end
end


  SUP=49;                 % Support of the largest filter (must be odd)
  SCALEX=sqrt(2).^[1:3];  % Sigma_{x} for the oriented filters
  NORIENT=6;              % Number of orientations

  NROTINV=12;
  NBAR=length(SCALEX)*NORIENT;
  NEDGE=length(SCALEX)*NORIENT;
  NF=NBAR+NEDGE+NROTINV;

  if cudaFlag == 0
    F=zeros(SUP,SUP,NF);
  else
    F=zeros(SUP,SUP,NF, 'gpuArray' );
  end

  hsup=(SUP-1)/2;
  [x,y]=meshgrid([-hsup:hsup],[hsup:-1:-hsup]);
  orgpts=[x(:) y(:)]';

  count=1;
  for scale=1:length(SCALEX),
    for orient=0:NORIENT-1,
      angle=pi*orient/NORIENT;  % Not 2pi as filters have symmetry
      c=cos(angle);s=sin(angle);
      rotpts=[c -s;s c]*orgpts;
      
      if cudaFlag == 0
        F(:,:,count)=makefilter(SCALEX(scale),0,1,rotpts,SUP);
        F(:,:,count+NEDGE)=makefilter(SCALEX(scale),0,2,rotpts,SUP);
      else
        F(:,:,count) = gpuArray( makefilter(SCALEX(scale),0,1,rotpts,SUP) );
        F(:,:,count+NEDGE) = gpuArray( makefilter(SCALEX(scale),0,2,rotpts,SUP) );
      end
      
      count=count+1;
    end;
  end;
  
  count=NBAR+NEDGE+1;
  SCALES=sqrt(2).^[1:4];
  
  if cudaFlag == 0
    for i=1:length(SCALES),
      F(:,:,count)   = normalise(fspecial('gaussian',SUP,SCALES(i)));
      F(:,:,count+1) = normalise(fspecial('log',SUP,SCALES(i)));
      F(:,:,count+2) = normalise(fspecial('log',SUP,3*SCALES(i)));
      count=count+3;
    end
  else
    for i=1:length(SCALES),
      F(:,:,count)   = gpuArray( normalise(fspecial('gaussian',SUP,SCALES(i))) );
      F(:,:,count+1) = gpuArray( normalise(fspecial('log',SUP,SCALES(i))) );
      F(:,:,count+2) = gpuArray( normalise(fspecial('log',SUP,3*SCALES(i))) );
      count=count+3;
    end
  end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f=makefilter(scale,phasex,phasey,pts,sup)
  gx=gauss1d(3*scale,0,pts(1,:),phasex);
  gy=gauss1d(scale,0,pts(2,:),phasey);
  f=normalise(reshape(gx.*gy,sup,sup));
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function g=gauss1d(sigma,mean,x,ord)
% Function to compute gaussian derivatives of order 0 <= ord < 3
% evaluated at x.

  x=x-mean;num=x.*x;
  variance=sigma^2;
  denom=2*variance;  
  g=exp(-num/denom)/(pi*denom)^0.5;
  switch ord,
    case 1, g=-g.*(x/variance);
    case 2, g=g.*((num-variance)/(variance^2));
  end;
return

function f=normalise(f), f=f-mean(f(:)); f=f/sum(abs(f(:))); return