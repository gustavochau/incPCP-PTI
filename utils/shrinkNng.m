function[u] = shrinkNgg(v, lambda, alpha, beta)

la = single( lambda*alpha );
lb = single( lambda*beta );

mask = single( abs(v) > (la+lb) );
Neg  = single( 2*((v > 0) - 0.5) );


u = mask.*( v.*(v - 2.0*la*Neg ) + la*la - lb*lb )./( mask.*(v - la*Neg) + 1.0 - mask);


return
