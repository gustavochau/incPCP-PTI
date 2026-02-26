function[H] = fbank2freq(h, Nrows, Ncols, cudaFlag)


if nargin < 4
  cudaFlag = 0;
end


[hr hc hn] = size(h);

if cudaFlag == 0

  H = zeros(Nrows+floor(hr/2), Ncols+floor(hc/2), hn) + j*zeros(Nrows+floor(hr/2), Ncols+floor(hc/2), hn);

  for k=1:hn
    % 
    H(:,:,k) = fft2( h(:,:,k), Nrows+floor(hr/2), Ncols+floor(hc/2));
  
  end
  
else
  H = zeros([Nrows+floor(hr/2), Ncols+floor(hc/2), hn], 'gpuArray') + 1i*zeros( [Nrows+floor(hr/2), Ncols+floor(hc/2), hn], 'gpuArray');

  for k=1:hn
    % 
    H(:,:,k) = fft2( gpuArray(h(:,:,k)), Nrows+floor(hr/2), Ncols+floor(hc/2));
  
  end
  
end


%   NOTE: check bsxfun for applying this FBank