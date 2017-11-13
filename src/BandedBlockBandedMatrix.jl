# Represents a block banded matrix with banded blocks
#   similar to BandedMatrix{BandedMatrix{T}}
struct BandedBlockBandedMatrix{T} <: AbstractBlockBandedMatrix{T}
    data::Matrix{T}
    block_sizes::BlockSizes{2}

    l::Int  # block lower bandwidth
    u::Int  # block upper bandwidth
    λ::Int  # sub lower bandwidth
    μ::Int  # sub upper bandwidth

    function BandedBlockBandedMatrix{T}(data::Matrix{T}, block_sizes::BlockSizes{2},
                                             l::Int, u::Int, λ::Int, μ::Int) where T
        n = block_sizes[1][end]-1 # number of rows
        if (size(data,1) ≠ λ+μ+1  && !(size(data,1) == 0 && -λ > μ)) ||
              (size(data,2) ≠ (l+u+1)*n && !(size(data,2) == 0 && -l > u))
              throw(ArgumentError("Data matrix must have number rows equal to number of bands"))
        end
        new{T}(data, block_sizes, l, u, λ, μ)
    end

    @inline BandedBlockBandedMatrix{T}(data::AbstractMatrix, block_sizes::BlockSizes{2},
                        l::Int, u::Int, λ::Int, μ::Int) where T =
        new{T}(Matrix{T}(data), block_sizes, l, u, λ, μ)

    BandedBlockBandedMatrix{T}(block_sizes::BlockSizes{2},
                        l::Int, u::Int, λ::Int, μ::Int) where T =
        BandedBlockBandedMatrix{T}(Matrix{T}(max(0, λ+μ+1), max(0,(l+u+1)*(block_sizes.cumul_sizes[2][end]-1))),
                                        block_sizes, l, u, λ, μ)
end


# Auxiliary outer constructors

@inline BandedBlockBandedMatrix(data::AbstractMatrix, block_sizes::BlockSizes{2},
                                         l::Int, u::Int, λ::Int, μ::Int) =
    BandedBlockBandedMatrix{eltype(data)}(data, block_sizes, l, u, λ, μ)

@inline BandedBlockBandedMatrix(data::AbstractMatrix, block_sizes::NTuple{2, Vector{Int}},
                                         lu::NTuple{2, Int}, λμ::NTuple{2, Int}) =
    BandedBlockBandedMatrix{eltype(data)}(data, BlockSizes(block_sizes...), lu..., λμ...)

@inline BandedBlockBandedMatrix(data::AbstractMatrix, block_sizes::NTuple{2, AbstractVector{Int}},
                                    lu::NTuple{2, Int}, λμ::NTuple{2, Int}) =
    BandedBlockBandedMatrix(data, Vector{Int}.(block_sizes), lu, λμ)


@inline BandedBlockBandedMatrix{T}(data::AbstractMatrix, block_sizes::NTuple{2, Vector{Int}},
                                         lu::NTuple{2, Int}, λμ::NTuple{2, Int}) where T =
    BandedBlockBandedMatrix{T}(data, BlockSizes(block_sizes...), lu..., λμ...)

@inline BandedBlockBandedMatrix{T}(data::AbstractMatrix, block_sizes::NTuple{2, AbstractVector{Int}},
                                    lu::NTuple{2, Int}, λμ::NTuple{2, Int}) where T =
    BandedBlockBandedMatrix{T}(data, Vector{Int}.(block_sizes), lu, λμ)


convert(::Type{BandedBlockBandedMatrix{T}}, B::BandedMatrix) where T =
    if isdiag(B)
        BandedBlockBandedMatrix{T}(copy(B.data),0,0,0,0,ones(Int,size(B,1)),ones(Int,size(B,2)))
    else
        BandedBlockBandedMatrix{T}(copy(B.data),0,0,B.l,B.u,[size(B,1)],[size(B,2)])
    end

convert(::Type{BandedBlockBandedMatrix}, B::BandedMatrix) = convert(BandedBlockBandedMatrix{eltype(B)}, B)



################################
# BandedBlockBandedMatrix Interface #
################################

blockbandwidth(A::BandedBlockBandedMatrix, i::Int) = ifelse(i==1, A.λ, A.μ)

isdiag(A::BandedBlockBandedMatrix) = A.λ == A.μ == A.l == A.u

################################
# AbstractBlockArray Interface #
################################

@inline nblocks(block_array::BandedBlockBandedMatrix) = nblocks(block_array.block_sizes)
@inline blocksize(block_array::BandedBlockBandedMatrix, i1::Int, i2::Int) = blocksize(block_array.block_sizes, (i1,i2))

@inline function getblock(A::BandedBlockBandedMatrix, K::Int, J::Int)
    @boundscheck blockcheckbounds(A, K, J)
    if -A.l ≤ J - K ≤ A.u
        convert(BandedMatrix, view(A, Block(K, J)))
    else
        zeroblock(A, K, J)
    end
end

# @inline function Base.getindex(block_arr::BlockArray{T,N}, blockindex::BlockIndex{N}) where {T,N}
#     @boundscheck checkbounds(block_arr.blocks, blockindex.I...)
#     @inbounds block = block_arr.blocks[blockindex.I...]
#     @boundscheck checkbounds(block, blockindex.α...)
#     @inbounds v = block[blockindex.α...]
#     return v
# end


###########################
# AbstractArray Interface #
###########################

# @inline function Base.similar(block_array::BandedBlockBandedMatrix{T}, ::Type{T2}) where {T,N,T2}
#     BlockArray(similar(block_array.blocks, Array{T2, N}), copy(block_array.block_sizes))
# end

Base.size(arr::BandedBlockBandedMatrix) =
    @inbounds return (arr.block_sizes[1][end] - 1, arr.block_sizes[2][end] - 1)


@inline function getindex(A::BandedBlockBandedMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    bi = global2blockindex(A.block_sizes, (i, j))
    @inbounds v = view(A, Block(bi.I))[bi.α...]
    return v
end

@inline function setindex!(A::BandedBlockBandedMatrix{T}, v, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    bi = global2blockindex(A.block_sizes, (i, j))
    V = view(A, Block(bi.I))
    @inbounds V[bi.α...] = convert(T, v)::T
    return v
end

############
# Indexing #
############

# function _check_setblock!(block_arr::BlockArray{T, N}, v, block::NTuple{N, Int}) where {T,N}
#     for i in 1:N
#         if size(v, i) != blocksize(block_arr.block_sizes, i, block[i])
#             throw(DimensionMismatch(string("tried to assign $(size(v)) array to ", blocksize(block_arr, block...), " block")))
#         end
#     end
# end
#
#
# @inline function setblock!(block_arr::BlockArray{T, N}, v, block::Vararg{Int, N}) where {T,N}
#     @boundscheck blockcheckbounds(block_arr, block...)
#     @boundscheck _check_setblock!(block_arr, v, block)
#     @inbounds block_arr.blocks[block...] = v
#     return block_arr
# end
#
# @propagate_inbounds function Base.setindex!(block_array::BlockArray{T, N}, v, block_index::BlockIndex{N}) where {T,N}
#     getblock(block_array, block_index.I...)[block_index.α...] = v
# end

########
# Misc #
########

# @generated function Base.Array(block_array::BlockArray{T, N, R}) where {T,N,R}
#     # TODO: This will fail for empty block array
#     return quote
#         block_sizes = block_array.block_sizes
#         arr = similar(block_array.blocks[1], size(block_array)...)
#         @nloops $N i i->(1:nblocks(block_sizes, i)) begin
#             block_index = @ntuple $N i
#             indices = globalrange(block_sizes, block_index)
#             arr[indices...] = getblock(block_array, block_index...)
#         end
#
#         return arr
#     end
# end
#
# @generated function Base.copy!(block_array::BlockArray{T, N, R}, arr::R) where {T,N,R <: AbstractArray}
#     return quote
#         block_sizes = block_array.block_sizes
#
#         @nloops $N i i->(1:nblocks(block_sizes, i)) begin
#             block_index = @ntuple $N i
#             indices = globalrange(block_sizes, block_index)
#             copy!(getblock(block_array, block_index...), arr[indices...])
#         end
#
#         return block_array
#     end
# end
#
# function Base.fill!(block_array::BlockArray, v)
#     for block in block_array.blocks
#         fill!(block, v)
#     end
# end


subblockbandwidths(A::BandedBlockBandedMatrix) = A.λ, A.μ
subblockbandwidth(A::BandedBlockBandedMatrix, k::Integer) = ifelse(k==1 , A.λ , A.μ)



##################
# BandedBlockBandedBlock
#
#   views of the blocks satisfy the BandedMatrix interface, and are memory-compatible
#   with BLASBandedMatrix.
##################

const BlockSlice1 = BlockSlice{Block{1,Int}}
const BandedBlockBandedBlock{T} = SubArray{T,2,BandedBlockBandedMatrix{T},Tuple{BlockSlice1,BlockSlice1},false}




######################################
# BandedMatrix interface  for Blocks #
######################################
@inline bandwidth(V::BandedBlockBandedBlock, k::Int) = ifelse(k == 1, parent(V).λ, parent(V).μ)



# gives the columns of parent(V).data that encode the block
blocks(V::BandedBlockBandedBlock)::Tuple{Int,Int} = first(first(parentindexes(V)).block.n),
                                                    first(last(parentindexes(V)).block.n)


function bbb_data_firstcol(V::BandedBlockBandedBlock)
    A = parent(V)
    K = first(first(parentindexes(V)).block.n)
    J_slice = last(parentindexes(V))
    J = first(J_slice.block.n)
    m = length(J_slice.indices)
    col1 = (A.block_sizes[2][J]-1)*(A.l+A.u+1) + (K-J + A.u)*m+1
end

function bbb_data_cols(V::BandedBlockBandedBlock)
    A = parent(V)
    K = first(first(parentindexes(V)).block.n)
    J_slice = last(parentindexes(V))
    J = first(J_slice.block.n)
    m = length(J_slice.indices)
    col1 = (A.block_sizes[2][J]-1)*(A.l+A.u+1) + (K-J + A.u)*m+1
    col1:col1+m-1
end




@inline function inbands_getindex(V::BandedBlockBandedBlock, k::Int, j::Int)
    A = parent(V)
    cols = bbb_data_cols(V)
    u = A.μ
    @inbounds return A.data[u + k - j + 1, cols[j]]
end

@inline function inbands_setindex!(V::BandedBlockBandedBlock{T}, v, k::Int, j::Int) where T
    A = parent(V)
    cols = bbb_data_cols(V)
    u = A.μ
    @inbounds A.data[u + k - j + 1, cols[j]] = convert(T, v)::T
    v
end

@propagate_inbounds function getindex(V::BandedBlockBandedBlock, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = blocks(V)
    if -A.l ≤ J-K ≤ A.u
        cols = bbb_data_cols(V)
        banded_getindex(view(A.data, :, cols), A.λ, A.μ, k, j)
    else
        zero(eltype(V))
    end
end

@propagate_inbounds function setindex!(V::BandedBlockBandedBlock, v, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = blocks(V)
    if -A.l ≤ J-K ≤ A.u
        cols = bbb_data_cols(V)
        banded_setindex!(view(A.data, :, cols), A.λ, A.μ, v, k, j)
    elseif iszero(v) # allow setindex for 0 datya
        v
    else
        throw(BandError(parent(V), J-K))
    end
end




function convert(::Type{BandedMatrix{T}}, V::BandedBlockBandedBlock) where {T}
    A = parent(V)
    cols = bbb_data_cols(V)
    K,J = blocks(V)
    BandedMatrix(Matrix{T}(view(A.data,:,cols)), size(V,1), A.λ, A.μ)
end

convert(::Type{BandedMatrix}, V::BandedBlockBandedBlock) = convert(BandedMatrix{eltype(V)}, V)





#############
# Linear algebra
#############


# BLAS structure
function Base.pointer(V::BandedBlockBandedBlock{T}) where {T<:BlasFloat}
    A = parent(parent(V))
    K,J = parentindexes(V)
    if K.K < J.K-A.u || K.K > J.K+A.l
        error("Cannot create pointer to zero blocks")
    end
    # column block K-J+A.u+1,J
    p = pointer(A.data)
    st = stride(A.data,2)
    sz = sizeof(T)
    col = bbb_data_firstcol(V)
    p+(col-1)*st*sz
end

@inline leadingdimension(V::BandedBlockBandedBlock) = stride(parent(V).data,2)
@inline blasstructure(::Type{BandedBlockBandedBlock{<:BlasFloat}}) = BlasStrided()

@banded BandedBlockBandedBlock
@banded_banded_linalg BandedBlockBandedBlock BandedSubBandedMatrix






function *(A::BandedBlockBandedMatrix{T},
           B::BandedBlockBandedMatrix{V}) where {T<:Number,V<:Number}
    Arows, Acols = A.block_sizes.cumul_sizes
    Brows, Bcols = B.block_sizes.cumul_sizes
    if Acols ≠ Brows
        # diagonal matrices can be converted
        if isdiag(B) && size(A,2) == size(B,1) == size(B,2)
            B = BandedBlockBandedMatrix(B.data, BlockSizes((Acols,Acols)), 0, 0, 0, 0)
        elseif isdiag(A) && size(A,2) == size(B,1) == size(A,1)
            A = BandedBlockBandedMatrix(A.data, BlockSizes((Brows,Brows)), 0, 0, 0, 0)
        else
            throw(DimensionMismatch("*"))
        end
    end
    n,m = size(A,1), size(B,2)

    A_mul_B!(BandedBlockBandedMatrix{promote_type(T,V)}(BlockSizes((Arows,Bcols)),
                                     A.l+B.l, A.u+B.u,
                                     A.λ+B.λ, A.μ+B.μ),
             A, B)
end
