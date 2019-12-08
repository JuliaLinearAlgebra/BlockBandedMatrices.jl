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
    BlockSizes(size.(A.diag, 1), size.(A.diag,2))
end    


# Block Tridiagonal
const BlockTridiagonal{T,VT<:Matrix{T}} = BlockMatrix{T,<:Tridiagonal{VT}}

BlockTridiagonal(A,B,C) = mortar(Tridiagonal(A,B,C))

function sizes_from_blocks(A::Tridiagonal, _) 
    # for k = 1:length(A.du)
    #     size(A.du[k],1) == sz[1][k] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
    #     size(A.du[k],2) == sz[2][k+1] || throw(ArgumentError("block sizes of upper diagonal inconsisent with diagonal"))
    #     size(A.dl[k],1) == sz[1][k+1] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
    #     size(A.dl[k],2) == sz[2][k] || throw(ArgumentError("block sizes of lower diagonal inconsisent with diagonal"))
    # end
    (size.(A.d, 1), size.(A.d,2))
end

blockbandwidths(A::BlockArray) = bandwidths(A.blocks)
isblockbanded(A::BlockArray) = isbanded(A.blocks)

@inline function getblock(block_arr::BlockTridiagonal{T,VT}, K::Int, J::Int) where {T,VT<:AbstractMatrix}
    @boundscheck blockcheckbounds(block_arr, K, J)
    abs(J-K) ≥ 2 && return convert(VT, Zeros{T}(length.(getindex.(axes(block_arr),(Block(K),Block(J))))...))
    block_arr.blocks[K,J]
end

checksquareblocks(A) = blockisequal(axes(A)...) || throw(DimensionMismatch("blocks are not square: block dimensions are $(axes(A))"))

for op in (:-, :+)
    @eval begin
        function $op(A::BlockTridiagonal, λ::UniformScaling) 
            checksquareblocks(A)
            mortar(Tridiagonal(A.blocks.dl, broadcast($op, A.blocks.d, Ref(λ)), A.blocks.du))
        end
        function $op(λ::UniformScaling, A::BlockTridiagonal) 
            checksquareblocks(A)
            mortar(Tridiagonal(A.blocks.dl, broadcast($op, Ref(λ), A.blocks.d), A.blocks.du))
        end
    end
end

function replace_in_print_matrix(A::BlockDiagonal, i::Integer, j::Integer, s::AbstractString)
    bi = findblockindex.(axes(A), (i,j))
    I,J = block.(bi)
    i,j = blockindex.(bi)
    Int(J-I) == 0 ? s : Base.replace_with_centered_mark(s)
end

function replace_in_print_matrix(A::BlockTridiagonal, i::Integer, j::Integer, s::AbstractString)
    bi = findblockindex.(axes(A), (i,j))
    I,J = block.(bi)
    i,j = blockindex.(bi)
    -1 ≤ Int(J-I) ≤ 1 ? s : Base.replace_with_centered_mark(s)
end