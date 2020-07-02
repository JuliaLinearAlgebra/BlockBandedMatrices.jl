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

bandeddata(P::PseudoBlockMatrix) = bandeddata(P.blocks)
bandwidths(P::PseudoBlockMatrix) = bandwidths(P.blocks)

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

@inline function getblock(block_arr::BlockBidiagonal{T,VT}, K::Int, J::Int) where {T,VT<:AbstractMatrix}
    @boundscheck blockcheckbounds(block_arr, K, J)
    l,u = blockbandwidths(block_arr)
    -l ≤ (J-K) ≤ u || return convert(VT, Zeros{T}(length.(getindex.(axes(block_arr),(Block(K),Block(J))))...))
    block_arr.blocks[K,J]
end

@inline function getblock(block_arr::BlockTridiagonal{T,VT}, K::Int, J::Int) where {T,VT<:AbstractMatrix}
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
            mortar(Tridiagonal(A.blocks.dl, broadcast($op, A.blocks.d, Ref(λ)), A.blocks.du))
        end
        function $op(λ::UniformScaling, A::BlockTridiagonal)
            checksquareblocks(A)
            mortar(Tridiagonal(broadcast($op,A.blocks.dl), broadcast($op, Ref(λ), A.blocks.d), broadcast($op,A.blocks.du)))
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
