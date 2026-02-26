function[x, l, loops] = projL1v5(b, c, myerr, nMaxIter, flag)
%  
%  Finds min 0.5*|| x - b ||_2^2  s.t.  || x ||_1 <= c
%  
%  
% Finds l s.t. || x ||_1 = c, where x = shrink(b,l)
%  
%  


if nargin < 5
  flag = 1;
end

%  strcmp(class(b), 'gpuArray')

s = sign(b);

bnorm = s'*b;
bAbs = s.*b;

bMax = max(abs(b));


xnorm = bnorm;
l = 0;

if bnorm <= c
  x = b;
  return;
end

% ===================================
% one iter


  sN = length(b);
  
  l0 = (xnorm+l*sN - c)/sN;  
  
  x = max(0, bAbs - l0);
  
  xnorm = sum(x);
    
  s = s.*(x>0);

%    -------------------------
  
  
  
  sN = s'*s;

  
  cotaInf = l0*sN/bMax;
  % cotaSup = sN;  
  
  gap = sN - cotaInf;
  
  alpha = 0.1;
  while alpha<=1,
    
    N = (cotaInf+alpha*sN)/(alpha+1);
    
    l = l0*(sN/(cotaInf+alpha*gap));
        
    x = max(0, bAbs - l);

    xnorm = sum(x);

%      [cotaInf, N, sN]
    
%      [l alpha]
    
    if (xnorm >= c)
      break;
    end
    
    alpha = alpha+0.1;
  end
  
  loops(1) = alpha;
  
%    -------------------------


for k = 1:nMaxIter

  if k>1
    sN = s'*s;
    l = (xnorm+l*sN - c)/sN;
    
    x = max(0, bAbs - l);  
    xnorm = sum(x);
    
  end

    

  
  if abs( xnorm - c )/bnorm  < myerr
    break;
  end

  
  s = s.*(x>0);
  
end

if flag == 1
  x = s.*x;
else
  x = s.*(abs(b)-l);
end


loops(2) = k;


