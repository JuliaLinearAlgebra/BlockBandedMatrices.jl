##
# Sparse BroadcastStyle
##
BroadcastStyle(::StructuredMatrixStyle{<:Diagonal}, ::BandedBlockBandedStyle) =
    BandedBlockBandedStyle()
BroadcastStyle(::BandedBlockBandedStyle, ::StructuredMatrixStyle{<:Diagonal}) =
    BandedBlockBandedStyle()


function blockbandwidths(P::BlockedMatrix{<:Any,<:Diagonal})
    blockisequal(axes(P,1),axes(P,2)) || throw(DimensionMismatch())
    (0,0)
end

blockbandwidths(::Diagonal) = (0,0)





blockbandwidths(A::BlockArray) = bandwidths(A.blocks)
isblockbanded(A::BlockArray) = isbanded(A.blocks)





###
# Zeros
###

blockbandwidths(::Zeros) = (-1,-1)
subblockbandwidths(::Zeros) = (-1,-1)


###
# DiagonalBlockMatrix
###

subblockbandwidths(::Diagonal) = (0,0)
bandedblockbandeddata(D::Diagonal) = permutedims(D.diag)

###
# Block-BandedMatrix
###



# fixed block sizes, we can figure out how far we encroach other blocks by looking at last column
function blockbandwidths(::AbstractBandedLayout, (a,b)::Tuple{AbstractBlockedUnitRange{<:Any,<:AbstractRange}, OneTo{Int}}, A)
    l,u = bandwidths(A)
    m = min(length(a), l + length(b))
    if u ≥ 0
        Int(findblock(a,m))-1,0
    else
        Int(findblock(a,m))-1,1-Int(findblock(a,min(length(a),1-u)))
    end
end




#####
# BlockKron
#####

isblockbanded(K::BlockKron) = isbanded(first(K.args))
isbandedblockbanded(K::BlockKron) = all(isbanded, K.args)
blockbandwidths(K::BlockKron) = bandwidths(first(K.args))
subblockbandwidths(K::BlockKron) = bandwidths(last(K.args))

_blockkron(::Tuple{Vararg{AbstractBandedLayout}}, A) = BandedBlockBandedMatrix(BlockKron(A...))


sublayout(::DiagonalLayout{L}, inds::Type{<:NTuple{2,BS}}) where {L,BS<:BlockSlice{<:BlockRange1}} = bandedblockbandedcolumns(sublayout(L(),Tuple{BS}))



##
# special for unitblocks
blockbandwidths(A::BlockedMatrix{<:Any,<:Any,<:NTuple{2,BlockedOneTo{Int,<:AbstractUnitRange{Int}}}}) = bandwidths(A.blocks)
blockbandwidths(A::BlockedMatrix{<:Any,<:Diagonal,<:NTuple{2,BlockedOneTo{Int,<:AbstractUnitRange{Int}}}}) = bandwidths(A.blocks)
subblockbandwidths(A::BlockedMatrix{<:Any,<:Any,<:NTuple{2,BlockedOneTo{Int,<:AbstractUnitRange{Int}}}}) = (0,0)


## BlockVector
subblockbandwidths(a::AbstractVector) = (blocksize(a,1)-1, 0)