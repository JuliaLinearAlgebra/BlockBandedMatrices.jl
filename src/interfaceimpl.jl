##
# Sparse BroadcastStyle
##
BroadcastStyle(::StructuredMatrixStyle{<:Diagonal}, ::BandedBlockBandedStyle) =
    BandedBlockBandedStyle()
BroadcastStyle(::BandedBlockBandedStyle, ::StructuredMatrixStyle{<:Diagonal}) =
    BandedBlockBandedStyle()


function blockbandwidths(P::PseudoBlockMatrix{<:Any,<:Diagonal})
    bs = blocksizes(P)
    cumulsizes(bs)[1] == cumulsizes(bs)[2] || throw(DimensionMismatch())
    (0,0)
end
bandwidths(P::PseudoBlockMatrix) = bandwidths(P.blocks)

BroadcastStyle(::Type{<:SubArray{<:Any,2,<:PseudoBlockMatrix{<:Any,<:Diagonal},
                                NTuple{2,BlockSlice1}}}) = BandedStyle()


isblockbanded(K::Kron{<:Any,2}) = isbanded(first(K.arrays))
isbandedblockbanded(K::Kron{<:Any,2}) = all(isbanded, K.arrays)
blockbandwidths(K::Kron{<:Any,2}) = bandwidths(first(K.arrays))
subblockbandwidths(K::Kron{<:Any,2}) = bandwidths(last(K.arrays))
function blocksizes(K::Kron{<:Any,2})
    A,B = K.arrays
    BlockSizes(Fill(size(B,1), size(A,1)), Fill(size(B,2), size(A,2)))
end

const SubKron{T,M1,M2,R1,R2} =
    SubArray{T,2,<:Kron{T,2,Tuple{M1,M2}},Tuple{BlockSlice{R1},BlockSlice{R2}}}


BroadcastStyle(::Type{<:SubKron{<:Any,<:Any,B,Block1,Block1}}) where B =
    BroadcastStyle(B)

@inline bandwidths(V::SubKron{<:Any,<:Any,<:Any,Block1,Block1}) =
    subblockbandwidths(parent(V))
