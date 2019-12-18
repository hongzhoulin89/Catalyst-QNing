function [r] = tworec(slist,ylist,rholist,q,r)

% Description: Apply two loop recursion to compute the descent
%              direction of L-BFGS

   if nargin < 5
      error( '5 parameters are required' ); 
   end
       
   [p_slist,n_slist] = size(slist); 
   [p_ylist,n_ylist] = size(ylist); 
   [p_rholist,n_rholist] = size(rholist); 

   if n_slist ~= n_ylist
       error('length of slist and ylist must be the same');
   elseif n_rholist ~= n_ylist
       error('length of rholist and ylist must be the same');
   elseif p_slist ~= p_ylist
       error('dimension of slist and ylist must be the same');
   end;

   if n_slist> 0
       alphalist = zeros(n_slist,1);
       for k = n_slist:-1:1
           alpha = rholist(k)*(slist(:,k)'*q);
           alphalist(k) = alpha;
           q = q - alpha*ylist(:,k); 
       end
       
       gamma = (slist(:,n_slist)'*ylist(:,n_ylist))/(ylist(:,n_ylist)'*ylist(:,n_ylist));
       r = gamma*q;
       for i = 1:n_slist
           beta = rholist(i)*(ylist(:,i)'*r);
           r = r + (alphalist(i)-beta)*slist(:,i);
       end
   end
       
       
       
       
       
end
