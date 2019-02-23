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




# Block Tridiagonal

function sizes_from_blocks(A::Tridiagonal{<:AbstractMatrix})
    sz = (size.(A.d, 1), size.(A.d,2))
    for k = 1:length(A.du)
        size(A.du[k],1) == sz[1][k] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
        size(A.du[k],2) == sz[2][k+1] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
        size(A.dl[k],1) == sz[1][k+1] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
        size(A.dl[k],2) == sz[2][k] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
    end
    sz
end

@inline function getblock(block_arr::BlockMatrix{T,<:Tridiagonal{VT}}, K::Int, J::Int) where {T,VT<:AbstractMatrix}
    @boundscheck blockcheckbounds(block_arr, K, J)
    abs(J-K) ≥ 2 && return convert(VT, Zeros{T}(blocksize(block_arr,(K,J))))
    block_arr.blocks[K,J]
end