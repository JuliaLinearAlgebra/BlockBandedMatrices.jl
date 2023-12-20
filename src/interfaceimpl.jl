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




const BlockDiagonal{T,VT<:Matrix{T}} = BlockMatrix{T,<:Diagonal{VT}}

BlockDiagonal(A) = mortar(Diagonal(A))

function sizes_from_blocks(A::Diagonal, _)
    # for k = 1:length(A.du)
    #     size(A.du[k],1) == sz[1][k] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
    #     size(A.du[k],2) == sz[2][k+1] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
    #     size(A.dl[k],1) == sz[1][k+1] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
    #     size(A.dl[k],2) == sz[2][k] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
    # end
    (size.(A.diag, 1), size.(A.diag,2))
end


# Block Bi/Tridiagonal
const BlockTridiagonal{T,VT<:Matrix{T}} = BlockMatrix{T,<:Tridiagonal{VT}}
const BlockBidiagonal{T,VT<:Matrix{T}} = BlockMatrix{T,<:Bidiagonal{VT}}

BlockTridiagonal(A,B,C) = mortar(Tridiagonal(A,B,C))
BlockBidiagonal(A, B, uplo) = mortar(Bidiagonal(A,B,uplo))

function sizes_from_blocks(A::Tridiagonal, _)
    # for k = 1:length(A.du)
    #     size(A.du[k],1) == sz[1][k] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
    #     size(A.du[k],2) == sz[2][k+1] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
    #     size(A.dl[k],1) == sz[1][k+1] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
    #     size(A.dl[k],2) == sz[2][k] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
    # end
    (size.(A.d, 1), size.(A.d,2))
end

function sizes_from_blocks(A::Bidiagonal, _)
    # for k = 1:length(A.du)
    #     size(A.du[k],1) == sz[1][k] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
    #     size(A.du[k],2) == sz[2][k+1] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
    #     size(A.dl[k],1) == sz[1][k+1] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
    #     size(A.dl[k],2) == sz[2][k] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
    # end
    (size.(A.dv, 1), size.(A.dv,2))
end

blockbandwidths(A::BlockArray) = bandwidths(A.blocks)
isblockbanded(A::BlockArray) = isbanded(A.blocks)

@inline function BlockArrays.viewblock(block_arr::BlockBidiagonal{T,VT}, KJ::Block{2}) where {T,VT<:AbstractMatrix}
    K,J = KJ.n
    @boundscheck blockcheckbounds(block_arr, K, J)
    l,u = blockbandwidths(block_arr)
    -l ≤ (J-K) ≤ u || return convert(VT, Zeros{T}(length.(getindex.(axes(block_arr),(Block(K),Block(J))))...))
    block_arr.blocks[K,J]
end

@inline function BlockArrays.viewblock(block_arr::BlockTridiagonal{T,VT}, KJ::Block{2}) where {T,VT<:AbstractMatrix}
    K,J = KJ.n
    @boundscheck blockcheckbounds(block_arr, K, J)
    abs(J-K) ≥ 2 && return convert(VT, Zeros{T}(length.(getindex.(axes(block_arr),(Block(K),Block(J))))...))
    block_arr.blocks[K,J]
end

checksquareblocks(A) = blockisequal(axes(A)...) || throw(DimensionMismatch("blocks are not square: block dimensions are $(axes(A))"))

for op in (:-, :+)
    @eval begin
        function $op(A::BlockDiagonal, λ::UniformScaling)
            checksquareblocks(A)
            mortar(Diagonal(broadcast($op, A.blocks.diag, Ref(λ))))
        end
        function $op(λ::UniformScaling, A::BlockDiagonal)
            checksquareblocks(A)
            mortar(Diagonal(broadcast($op, Ref(λ), A.blocks.diag)))
        end

        function $op(A::BlockTridiagonal, λ::UniformScaling)
            checksquareblocks(A)
            mortar(Tridiagonal(broadcast($op, A.blocks.dl, Ref(0λ)),
                               broadcast($op, A.blocks.d, Ref(λ)),
                               broadcast($op, A.blocks.du, Ref(0λ))))
        end
        function $op(λ::UniformScaling, A::BlockTridiagonal)
            checksquareblocks(A)
            mortar(Tridiagonal(broadcast($op, Ref(0λ), A.blocks.dl),
                               broadcast($op, Ref(λ), A.blocks.d),
                               broadcast($op, Ref(0λ), A.blocks.du)))
        end
        function $op(A::BlockBidiagonal, λ::UniformScaling)
            checksquareblocks(A)
            mortar(Bidiagonal(broadcast($op, A.blocks.dv, Ref(λ)), A.blocks.ev, A.blocks.uplo))
        end
        function $op(λ::UniformScaling, A::BlockBidiagonal)
            checksquareblocks(A)
            mortar(Bidiagonal(broadcast($op, Ref(λ), A.blocks.dv), broadcast($op,A.blocks.ev), A.blocks.uplo))
        end
    end
end


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
