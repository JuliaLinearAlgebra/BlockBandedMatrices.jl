isblockbanded(K::Kron{<:Any,2}) = isbanded(first(K.arrays))
isbandedblockbanded(K::Kron{<:Any,2}) = all(isbanded, K.arrays)
blockbandwidths(K::Kron{<:Any,2}) = bandwidths(first(K.arrays))
subblockbandwidths(K::Kron{<:Any,2}) = bandwidths(last(K.arrays))
function block_sizes(K::Kron{<:Any,2})
    A,B = K.arrays
    BlockSizes(Fill(size(B,1), size(A,1)), Fill(size(B,2), size(A,2)))
end
