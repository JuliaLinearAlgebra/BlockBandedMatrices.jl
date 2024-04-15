##
# Sparse BroadcastStyle
##
BroadcastStyle(::StructuredMatrixStyle{<:Diagonal}, ::BandedBlockBandedStyle) =
    BandedBlockBandedStyle()
BroadcastStyle(::BandedBlockBandedStyle, ::StructuredMatrixStyle{<:Diagonal}) =
    BandedBlockBandedStyle()


function blockbandwidths(P::PseudoBlockMatrix{<:Any,<:Diagonal})
    blockisequal(axes(P,1),axes(P,2)) || throw(DimensionMismatch())
    (0,0)
end

blockbandwidths(::Diagonal) = (0,0)


BroadcastStyle(::Type{<:SubArray{<:Any,2,<:PseudoBlockMatrix{<:Any,<:Diagonal},
                                <:Tuple{<:BlockSlice1,<:BlockSlice1}}}) = BandedStyle()




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

sublayout(::DiagonalLayout{L}, inds::Type{<:NTuple{2,BS}}) where {L,BS<:BlockSlice{<:BlockRange1}} = bandedblockbandedcolumns(sublayout(L(),Tuple{BS}))
subblockbandwidths(::Diagonal) = (0,0)
bandedblockbandeddata(D::Diagonal) = permutedims(D.diag)

###
# Block-BandedMatrix
###



# fixed block sizes, we can figure out how far we encroach other blocks by looking at last column
function blockbandwidths(::AbstractBandedLayout, (a,b)::Tuple{BlockedUnitRange{<:AbstractRange}, OneTo{Int}}, A)
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
