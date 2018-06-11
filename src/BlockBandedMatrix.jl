####
# BlockBandedMatrix memory layout

struct BlockBandedLayout <: AbstractBlockBandedLayout end


#### Routines for BandedSizes
function bb_blockstarts(b_size, l, u)
    N,M = nblocks(b_size)
    b_start = BandedMatrix{Int}(uninitialized, N, M, l, u)
    -l > u && return b_start

    ind_shift = 0
    for J = 1:M
        KR = max(1,J-u):min(J+l,N)
        if !isempty(KR)
            b_start[KR,J] .= ind_shift .+ view(b_size.cumul_sizes[1],KR) .- b_size.cumul_sizes[1][KR[1]] .+ 1

            num_rows = b_size[1,KR[end]+1]-b_size[1,KR[1]]
            num_cols = blocksize(b_size, 2, J)
            ind_shift += num_rows*num_cols
        end
    end

    b_start
end

function bb_blockstrides(b_size, l, u)
    N, M = nblocks(b_size)
    b_strides = Vector{Int}(M)
    for J=1:M
        KR = max(1,J-u):min(J+l,N)
        if !isempty(KR)
            b_strides[J] = b_size[1,KR[end]+1]-b_size[1,KR[1]]
        else
            b_strides[J] = 0
        end
    end
    b_strides
end

struct BlockBandedSizes
    block_sizes::BlockSizes{2}
    block_starts::BandedMatrix{Int} # gives index where the blocks start
    block_strides::Vector{Int} # gives stride to next block for J-th column
end

BlockBandedSizes(b_size::BlockSizes{2}, l, u) =
    BlockBandedSizes(b_size, bb_blockstarts(b_size, l, u), bb_blockstrides(b_size, l, u))

BlockBandedSizes(rows::AbstractVector{Int}, cols::AbstractVector{Int}, l, u) =
    BlockBandedSizes(BlockSizes(rows,cols), l, u)


blockbandwidths(bs::BlockBandedSizes) = bandwidths(bs.block_starts)
blockbandwidth(bs::BlockBandedSizes, i::Int) = bandwidth(bs.block_starts, i)

for Func in (:nblocks, :getindex, :blocksize, :global2blockindex, :unblock, :size, :globalrange)
    @eval begin
        $Func(B::BlockBandedSizes) = $Func(B.block_sizes)
        $Func(B::BlockBandedSizes, k) = $Func(B.block_sizes, k)
        $Func(B::BlockBandedSizes, k, j) = $Func(B.block_sizes, k, j)
    end
end

function bb_numentries(B::BlockBandedSizes)
    b_size, l, u = B.block_sizes, B.block_starts.l, B.block_starts.u
    numentries = 0
    N = nblocks(b_size,1)
    for J = 1:nblocks(b_size,2),
        KR = colrange(B.block_starts, J)
        num_rows = b_size[1,KR[end]+1] - b_size[1,KR[1]]
        num_cols = blocksize(b_size, 2, J)
        numentries += num_rows*num_cols
    end
    numentries
end


function _BandedBlockMatrix end


#  A block matrix where only the bands are nonzero
#   isomorphic to BandedMatrix{Matrix{T}}
struct BlockBandedMatrix{T} <: AbstractBlockBandedMatrix{T}
    data::Vector{T}
    block_sizes::BlockBandedSizes

    l::Int  # block lower bandwidth
    u::Int  # block upper bandwidth

    global function _BlockBandedMatrix(data::Vector{T}, block_sizes::BlockBandedSizes) where T
        new{T}(data, block_sizes, blockbandwidth(block_sizes,1), blockbandwidth(block_sizes,2))
    end
end

# Auxiliary outer constructors
@inline _BlockBandedMatrix(data::AbstractVector, dims::NTuple{2, AbstractVector{Int}}, lu::NTuple{2, Int}) =
    _BlockBandedMatrix(data, BlockBandedSizes(dims..., lu...))

@inline BlockBandedMatrix{T}(::Uninitialized, block_sizes::BlockBandedSizes) where T =
    _BlockBandedMatrix(Vector{T}(uninitialized, bb_numentries(block_sizes)), block_sizes)

"""
    BlockBandedMatrix{T}(uninitialized, (rows, cols), (l, u))

returns an uninitialized `sum(rows)`×`sum(cols)` block-banded matrix `A`
of type `T` with block-bandwidths `(l,u)` and where `A[Block(K,J)]`
is a `Matrix{T}` of size `rows[K]`×`cols[J]`.
"""
BlockBandedMatrix

@inline BlockBandedMatrix{T}(::Uninitialized, dims::NTuple{2, AbstractVector{Int}}, lu::NTuple{2, Int}) where T =
    BlockBandedMatrix{T}(uninitialized, BlockBandedSizes(dims..., lu...))



function BlockBandedMatrix{T}(Z::Zeros, block_sizes::BlockBandedSizes) where T
   if size(Z) ≠ size(block_sizes)
       throw(DimensionMismatch("Size of input $(size(Z)) must be consistent with $(size(block_sizes))"))
   end
   _BlockBandedMatrix(zeros(T, bb_numentries(block_sizes)), block_sizes)
end

function BlockBandedMatrix{T}(A::AbstractMatrix, block_sizes::BlockBandedSizes) where T
    ret = BlockBandedMatrix(Zeros{T}(size(A)), block_sizes)
    for J = Block.(1:nblocks(ret, 2)), K = blockcolrange(ret, Int(J))
        kr, jr = globalrange(block_sizes, (Int(K), Int(J)))
        view(ret, K, J) .= view(A, kr, jr)
    end
    ret
end

function BlockBandedMatrix{T}(E::Eye, block_sizes::BlockBandedSizes) where T
    if size(E) ≠ size(block_sizes)
        throw(DimensionMismatch("Size of input $(size(E)) must be consistent with $(sum.(dims))"))
    end
    ret = BlockBandedMatrix(Zeros{T}(size(E)), block_sizes)
    ret[diagind(ret)] = one(T)
    ret
end

function BlockBandedMatrix{T}(A::UniformScaling, block_sizes::BlockBandedSizes) where T
    ret = BlockBandedMatrix(Zeros{T}(size(block_sizes)), block_sizes)
    ret[diagind(ret)] = convert(T, A.λ)
    ret
end

BlockBandedMatrix(A::Union{AbstractMatrix,UniformScaling},
                        block_sizes::BlockBandedSizes) = BlockBandedMatrix{eltype(A)}(A, block_sizes)

BlockBandedMatrix{T}(A::Union{AbstractMatrix,UniformScaling},
                     dims::NTuple{2,AbstractVector{Int}}, lu::NTuple{2,Int}) where T =
    BlockBandedMatrix{T}(A, BlockBandedSizes(dims..., lu...))


BlockBandedMatrix(A::Union{AbstractMatrix,UniformScaling},
                        dims::NTuple{2, AbstractVector{Int}},
                        lu::NTuple{2,Int}) = BlockBandedMatrix{eltype(A)}(A, dims, lu)


function convert(::Type{BlockBandedMatrix}, A::AbstractMatrix)
    @assert isblockbanded(A)
    BlockBandedMatrix(A, BlockBandedSizes(A.block_sizes))
end


################################
# BlockBandedMatrix Interface #
################################

MemoryLayout(::BlockBandedMatrix) = BlockBandedLayout()
blockbandwidth(A::BlockBandedMatrix, i::Int) = ifelse(i==1, A.l, A.u)


################################
# AbstractBlockArray Interface #
################################

@inline nblocks(block_array::BlockBandedMatrix) = nblocks(block_array.block_sizes)
@inline blocksize(block_array::BlockBandedMatrix, i1::Int, i2::Int) = blocksize(block_array.block_sizes, (i1,i2))

zeroblock(A::BlockBandedMatrix, K::Int, J::Int) =
    Matrix(Zeros{eltype(A)}(blocksize(A, K, J)))

@inline function getblock(A::BlockBandedMatrix, K::Int, J::Int)
    @boundscheck blockcheckbounds(A, K, J)
    if -A.l ≤ J - K ≤ A.u
        convert(Matrix, view(A, Block(K, J)))
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

# @inline function Base.similar(block_array::BlockBandedMatrix{T}, ::Type{T2}) where {T,N,T2}
#     BlockArray(similar(block_array.blocks, Array{T2, N}), copy(block_array.block_sizes))
# end

Base.size(arr::BlockBandedMatrix) =
    @inbounds return (arr.block_sizes.block_sizes[1][end] - 1, arr.block_sizes.block_sizes[2][end] - 1)


@inline function getindex(A::BlockBandedMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    bi = global2blockindex(A.block_sizes, (i, j))
    @inbounds v = view(A, Block(bi.I))[bi.α...]
    return v
end

@inline function setindex!(A::BlockBandedMatrix{T}, v, i::Int, j::Int) where T
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
@inline function setblock!(A::BlockBandedMatrix, v, K::Int, J::Int)
    @boundscheck blockcheckbounds(A, K, J)

    V = view(A, Block(K), Block(J))
    V .= v
    return A
end

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


##################
# BlockBandedBlock
#
#   views of the blocks satisfy the Matrix interface, and are memory-compatible
#   with StridedMatrix.
##################

const BlockBandedBlock{T} = SubArray{T,2,BlockBandedMatrix{T},Tuple{BlockSlice1,BlockSlice1},false}




# gives the columns of parent(V).data that encode the block
blocks(V::BlockBandedBlock)::Tuple{Int,Int} = first(first(parentindexes(V)).block.n),
                                                    first(last(parentindexes(V)).block.n)

######################################
# Matrix interface  for Blocks #
######################################


MemoryLayout(::BlockBandedBlock) = ColumnMajor()

function Base.unsafe_convert(::Type{Ptr{T}}, V::BlockBandedBlock{T}) where T
    A = parent(V)
    K,J = blocks(V)
    Base.unsafe_convert(Ptr{T}, A.data) + sizeof(T)*(A.block_sizes.block_starts[K,J]-1)
end

strides(V::BlockBandedBlock) = (1,parent(V).block_sizes.block_strides[blocks(V)[2]])

@propagate_inbounds function getindex(V::BlockBandedBlock, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = blocks(V)
    if -A.l ≤ J-K ≤ A.u
        b_start = A.block_sizes.block_starts[K,J]
        b_stride = A.block_sizes.block_strides[J]
        A.data[b_start + k-1 + (j-1)*b_stride ]
    else
        zero(eltype(V))
    end
end

@propagate_inbounds function setindex!(V::BlockBandedBlock, v, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = blocks(V)
    if -A.l ≤ J-K ≤ A.u
        b_start = A.block_sizes.block_starts[K,J]
        b_stride = A.block_sizes.block_strides[J]
        A.data[b_start + k-1 + (j-1)*b_stride ] = v
    elseif iszero(v) # allow setindex for 0 datya
        v
    else
        throw(BandError(A, J-K))
    end
end
