function [z] = wthresh(w,SORH,T)
%     Soft or hard thresholding
%     returns the soft (if SORH = 's') or hard (if SORH = 'h') thresholding
%
%     Inputs:
%       w     Vector to threshold
%       SORH  'h' or 's'
%       T     Threshold
%     Outputs:  
%       z     


       if nargin < 3
          error( '3 parameters are required' ); 
       end
       if (SORH ~= 'h' &  SORH ~= 's') then
          error(' SORH must be either h or s');
       end;

    if SORH=='h'
      z   = w .* (abs(w) > T);
    else
      res = (abs(w) - T);
      res = (res + abs(res))/2;
      z   = sign(w).*res;
    end;
end
