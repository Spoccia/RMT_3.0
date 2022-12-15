"""
This  function has the aim to produce a cutted clusters  near to a central variate
output is a vector of  0-1 on the base if the varuiate  have to be included in the neighborhood of the 
central variate or not
function H = normalize_12132015(H)
Ht = H';
for i=1:size(H,1) % = N
    if(sum(H(i,:))>0)
        H(i,:) = H(i,:)/sum(H(i,:));
    end
end
end

"""
import numpy as np

def normalize_dpd(H):
    ht = H.traspose()
    tmp =  np.sum(ht, axis=0)
    
