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
end

# Auxiliary outer constructors

@inline BandedBlockBandedMatrix(data::Matrix, block_sizes::BlockSizes{2},
                                         lu::Int, u::Int, λ::Int, μ::Int) =
    BandedBlockBandedMatrix{eltype(data)}(data, block_sizes, l, u, λ, μ)

@inline BandedBlockBandedMatrix(data::Matrix, block_sizes::NTuple{2, Vector{Int}},
                                         lu::NTuple{2, Int}, λμ::NTuple{2, Int}) =
    BandedBlockBandedMatrix{eltype(data)}(data, BlockSizes(block_sizes...), lu..., λμ...)

@inline BandedBlockBandedMatrix(data::Matrix, block_sizes::NTuple{2, AbstractVector{Int}},
                                    lu::NTuple{2, Int}, λμ::NTuple{2, Int}) =
    BandedBlockBandedMatrix(data, Vector{Int}.(block_sizes), lu, λμ)


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


@inline function Base.getindex(A::BandedBlockBandedMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    bi = global2blockindex(A.block_sizes, (i, j))
    @inbounds v = view(A, Block(bi.I))[bi.α...]
    return v
end

# @inline function Base.setindex!(block_arr::BlockArray{T, N}, v, i::Vararg{Int, N}) where {T,N}
#     @boundscheck checkbounds(block_arr, i...)
#     @inbounds block_arr[global2blockindex(block_arr.block_sizes, i)] = v
#     return block_arr
# end

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




##################
# BandedMatrix interface
##################

isbanded(::BandedBlockBandedBlock) = true

@inline leadingdimension(V::BandedBlockBandedBlock) = stride(parent(V).data,2)
@inline bandwidth(V::BandedBlockBandedBlock, k::Int) = ifelse(k == 1, parent(V).λ, parent(V).μ)



# gives the columns of parent(V).data that encode the block
blocks(V::BandedBlockBandedBlock)::Tuple{Int,Int} = first(first(parentindexes(V)).block.n),
                                                    first(last(parentindexes(V)).block.n)
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
