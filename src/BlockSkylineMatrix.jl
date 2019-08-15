checkbandwidths(N, M, l::AbstractVector{Int}, u::AbstractVector{Int}) =
    M == 1 || (length(u) == M && length(l) == M) ||
    throw(DimensionMismatch("For a matrix of $(N)×$(M) blocks, $(M) lower and upper column bandwidths are required"))

#### Routines for BandedSizes
function bb_blockstarts(b_size, l::AbstractVector{Int}, u::AbstractVector{Int})
    N,M = nblocks(b_size)
    L,U = maximum(l), maximum(u)
    b_start = BandedMatrix{Int}(undef, N, M, L, U)
    -L > U && return b_start

    checkbandwidths(N, M, l, u)

    ind_shift = 0
    for J = 1:M
        KR = max(1,J-u[J]):min(J+l[J],N)
        if !isempty(KR)
            b_start[KR,J] .= ind_shift .+ view(cumulsizes(b_size,1),KR) .- cumulsizes(b_size,1)[KR[1]] .+ 1

            num_rows = cumulsizes(b_size,1,KR[end]+1)-cumulsizes(b_size,1,KR[1])
            num_cols = blocksize(b_size, 2, J)
            ind_shift += num_rows*num_cols
        end
    end

    b_start
end

bb_blockstarts(b_size, l::Integer, u::Integer) = bb_blockstarts(b_size, Fill(l, nblocks(b_size,2)), Fill(u, nblocks(b_size,2)))

function bb_blockstrides(b_size, l::AbstractVector{Int}, u::AbstractVector{Int})
    N, M = nblocks(b_size)
    L,U = maximum(l), maximum(u)
    checkbandwidths(N, M, l, u)
    b_strides = Vector{Int}(undef, M)
    for J=1:M
        KR = max(1,J-u[J]):min(J+l[J],N)
        if !isempty(KR)
            b_strides[J] = cumulsizes(b_size,1,KR[end]+1)-cumulsizes(b_size,1,KR[1])
        else
            b_strides[J] = 0
        end
    end
    b_strides
end

bb_blockstrides(b_size, l::Integer, u::Integer) = bb_blockstrides(b_size, Fill(l, nblocks(b_size,1)), Fill(u, nblocks(b_size,2)))

struct BlockSkylineSizes{BS<:AbstractBlockSizes{2}, LL<:AbstractVector{Int}, UU<:AbstractVector{Int}, BStarts, BStrides} <: AbstractBlockSizes{2}
    block_sizes::BS
    block_starts::BStarts # gives index where the blocks start, usually a BandedMatrix{Int}
    block_strides::BStrides # gives stride to next block for J-th column, usually a Vector{Int}
    l::LL
    u::UU
end

BlockSkylineSizes(b_size::BlockSizes{2}, l::AbstractVector{Int}, u::AbstractVector{Int}) =
    BlockSkylineSizes(b_size, bb_blockstarts(b_size, l, u), bb_blockstrides(b_size, l, u), l, u)

BlockSkylineSizes(rows::AbstractVector{Int}, cols::AbstractVector{Int}, l::AbstractVector{Int}, u::AbstractVector{Int}) =
    BlockSkylineSizes(BlockSizes(rows,cols), l, u)

const BlockBandedSizes = BlockSkylineSizes{DefaultBlockSizes{2}, Fill{Int,1,Tuple{OneTo{Int}}}, Fill{Int,1,Tuple{OneTo{Int}}}, 
                                            BandedMatrix{Int,Matrix{Int},OneTo{Int}}, Vector{Int}}


BlockBandedSizes(b_size::BlockSizes{2}, l::Int, u::Int) =
    BlockSkylineSizes(b_size, Fill(l, nblocks(b_size,2)), Fill(u, nblocks(b_size,2)))
BlockBandedSizes(rows::AbstractVector{Int}, cols::AbstractVector{Int}, l::Int, u::Int) =
    BlockSkylineSizes(rows, cols, Fill(l, length(cols)), Fill(u, length(cols)))

colblockbandwidths(bs::BlockSkylineSizes) = (bs.l, bs.u)

blockstart(block_sizes::BlockSkylineSizes, K, J) = block_sizes.block_starts[K,J]
blockstride(block_sizes::BlockSkylineSizes, J) = block_sizes.block_strides[J]

blockstart(A::AbstractMatrix, K, J) = blockstart(blocksizes(A),K,J)
blockstride(A::AbstractMatrix, J) = blockstride(blocksizes(A),J)

blockbandwidths(bs::BlockSkylineSizes) = maximum.(colblockbandwidths(bs))
blockbandwidth(bs::BlockSkylineSizes, i::Int) = blockbandwidths(bs)[i]


==(A::BlockSkylineSizes, B::BlockSkylineSizes) = A.block_sizes == B.block_sizes && A.block_starts == B.block_starts &&
    A.l == B.l && A.u == B.u

cumulsizes(B::BlockSkylineSizes) = cumulsizes(B.block_sizes)

colrange(B::BlockSkylineSizes, J::Integer) = max(1, J-B.u[J]):min(nblocks(B.block_sizes,1), J+B.l[J])

function bb_numentries(B::BlockSkylineSizes)
    b_size = B.block_sizes
    numentries = 0
    N = nblocks(b_size,1)
    for J = 1:nblocks(b_size,2),
        KR = colrange(B, J)
        num_rows = cumulsizes(b_size,1,KR[end]+1) - cumulsizes(b_size,1,KR[1])
        num_cols = blocksize(b_size, 2, J)
        numentries += num_rows*num_cols
    end
    numentries
end


function _BandedBlockMatrix end


#  A block matrix where only the bands are nonzero
#   isomorphic to BandedMatrix{Matrix{T}}
struct BlockSkylineMatrix{T, DATA<:AbstractVector{T}, BS<:AbstractBlockSizes{2}} <: AbstractBlockBandedMatrix{T}
    data::DATA
    block_sizes::BS

    global function _BlockSkylineMatrix(data::DATA, block_sizes::BS) where {T,DATA<:AbstractVector{T}, BS<:AbstractBlockSizes{2}}
        new{T,DATA,BS}(data, block_sizes)
    end
end

const BlockBandedMatrix{T} = BlockSkylineMatrix{T, Vector{T}, BlockBandedSizes}

# Auxiliary outer constructors
@inline _BlockBandedMatrix(data::AbstractVector, bs::BlockBandedSizes) =
    _BlockSkylineMatrix(data, bs)

@inline _BlockBandedMatrix(data::AbstractVector, (kr,jr)::NTuple{2, AbstractVector{Int}}, (l,u)::NTuple{2, Int}) =
    _BlockBandedMatrix(data, BlockBandedSizes(kr,jr, l,u))

@inline BlockSkylineMatrix{T}(::UndefInitializer, block_sizes::BlockSkylineSizes) where T =
    _BlockSkylineMatrix(Vector{T}(undef, bb_numentries(block_sizes)), block_sizes)

@inline BlockBandedMatrix{T}(::UndefInitializer, block_sizes::BlockBandedSizes) where T =
    _BlockSkylineMatrix(Vector{T}(undef, bb_numentries(block_sizes)), block_sizes)

@inline BlockBandedMatrix{T}(::UndefInitializer, block_sizes::BlockSizes, (l,u)::NTuple{2, Int}) where T =
    BlockSkylineMatrix{T}(undef, BlockBandedSizes(block_sizes,l,u))

@inline BlockSkylineMatrix{T}(::UndefInitializer, block_sizes::BlockSizes, (l,u)::NTuple{2, AbstractVector{Int}}) where T =
    BlockSkylineMatrix{T}(undef, BlockSkylineSizes(block_sizes,l,u))

"""
    BlockSkylineMatrix{T,LL,UU}(undef, (rows, cols), (l::LL, u::UU))

returns an undef `sum(rows)`×`sum(cols)` block-banded matrix `A`
of type `T` with block-bandwidths `(l,u)` and where `A[Block(K,J)]`
is a `Matrix{T}` of size `rows[K]`×`cols[J]`.

`(l,u)` may be integers for constant bandwidths or integer vectors of
lengths `rows` and `cols`, respectively, for ragged bands.
"""
BlockSkylineMatrix

@inline BlockBandedMatrix{T}(::UndefInitializer, dims::NTuple{2, AbstractVector{Int}}, lu::NTuple{2, Int}) where T =
    BlockSkylineMatrix{T}(undef, BlockBandedSizes(dims..., lu...))

@inline BlockSkylineMatrix{T}(::UndefInitializer, dims::NTuple{2, AbstractVector{Int}}, lu::NTuple{2, AbstractVector{Int}}) where T =
    BlockSkylineMatrix{T}(undef, BlockSkylineSizes(dims..., lu...))

function BlockSkylineMatrix{T}(A::AbstractMatrix, block_sizes::BlockSkylineSizes) where T
    ret = BlockSkylineMatrix(Zeros{T}(size(A)), block_sizes)
    for J = Block.(1:nblocks(ret, 2)), K = blockcolrange(ret, Int(J))
        kr, jr = globalrange(block_sizes, (Int(K), Int(J)))
        view(ret, K, J) .= view(A, kr, jr)
    end
    ret
end

function BlockSkylineMatrix{T}(A::AbstractBlockBandedMatrix, block_sizes::BlockSkylineSizes) where T
    ret = BlockSkylineMatrix(Zeros{T}(size(A)), block_sizes)
    block_sizes == blocksizes(A) || throw(ArgumentError())
    for J = Block.(1:nblocks(ret, 2)), K = blockcolrange(ret, Int(J))
        view(ret, K, J) .= view(A, K, J)
    end
    ret
end

BlockBandedMatrix{T}(A::AbstractMatrix, block_sizes::BlockBandedSizes) where T =
    BlockSkylineMatrix{T}(A, block_sizes)

##
# Special cases
##

function BlockSkylineMatrix{T}(Z::Zeros, block_sizes::BlockSkylineSizes) where T
    if size(Z) ≠ size(block_sizes)
        throw(DimensionMismatch("Size of input $(size(Z)) must be consistent with $(size(block_sizes))"))
    end
    _BlockSkylineMatrix(zeros(T, bb_numentries(block_sizes)), block_sizes)
end


function BlockSkylineMatrix{T}(E::Eye, block_sizes::BlockSkylineSizes) where T
    if size(E) ≠ size(block_sizes)
        throw(DimensionMismatch("Size of input $(size(E)) must be consistent with $(sum.(dims))"))
    end
    ret = BlockSkylineMatrix(Zeros{T}(size(E)), block_sizes)
    ret[diagind(ret)] .= one(T)
    ret
end

function BlockSkylineMatrix{T}(A::UniformScaling, block_sizes::BlockSkylineSizes) where T
    ret = BlockSkylineMatrix(Zeros{T}(size(block_sizes)), block_sizes)
    ret[diagind(ret)] .= convert(T, A.λ)
    ret
end

BlockSkylineMatrix(A::Union{AbstractMatrix,UniformScaling},
                        block_sizes::BlockSkylineSizes) = BlockSkylineMatrix{eltype(A)}(A, block_sizes)
BlockBandedMatrix(A::Union{AbstractMatrix,UniformScaling},
                        block_sizes::BlockBandedSizes) = BlockBandedMatrix{eltype(A)}(A, block_sizes)


BlockSkylineMatrix{T}(A::Union{AbstractMatrix,UniformScaling},
                           dims::NTuple{2,AbstractVector{Int}}, lu::NTuple{2,AbstractVector{Int}}) where T =
                               BlockSkylineMatrix{T}(A, BlockSkylineSizes(dims..., lu...))

BlockBandedMatrix{T}(A::Union{AbstractMatrix,UniformScaling},
                           dims::NTuple{2,AbstractVector{Int}}, lu::NTuple{2,Int}) where T =
                               BlockSkylineMatrix{T}(A, BlockBandedSizes(dims..., lu...))

BlockSkylineMatrix(A::Union{AbstractMatrix,UniformScaling},
                        dims::NTuple{2, AbstractVector{Int}},
                        lu::NTuple{2,AbstractVector{Int}}) = BlockSkylineMatrix{eltype(A)}(A, dims, lu)
BlockBandedMatrix(A::Union{AbstractMatrix,UniformScaling},
                        dims::NTuple{2, AbstractVector{Int}},
                        lu::NTuple{2,Int}) = BlockBandedMatrix{eltype(A)}(A, dims, lu)

BlockBandedMatrix(A::BlockBandedMatrix, lu::NTuple{2,Int}) = BlockBandedMatrix(A, BlockBandedSizes(blocksizes(A).block_sizes, lu...))

function convert(::Type{BlockSkylineMatrix}, A::AbstractMatrix)
    @assert isblockbanded(A)
    block_sizes = convert(BlockSkylineSizes, A.block_sizes)

    ret = BlockSkylineMatrix{eltype(A)}(undef, block_sizes)
    for J = Block.(1:nblocks(ret, 2)), K = blockcolrange(ret, Int(J))
        view(ret, K, J) .= view(A, K, J)
    end
    ret
end

function convert(::Type{BlockBandedMatrix}, A::AbstractMatrix)
    @assert isblockbanded(A)
    convert(BlockSkylineMatrix, A)
end


BlockSkylineMatrix(A::AbstractMatrix) = convert(BlockSkylineMatrix, A)
BlockBandedMatrix(A::AbstractMatrix) = convert(BlockBandedMatrix, A)

similar(A::BlockSkylineMatrix, T::Type=eltype(A), bs::BlockSkylineSizes=blocksizes(A)) =
    BlockSkylineMatrix{T}(undef, bs)

################################
# BlockSkylineMatrix Interface #
################################

MemoryLayout(::Type{<:BlockSkylineMatrix}) = BlockBandedColumnMajor()
colblockbandwidths(A::BlockSkylineMatrix) = (A.block_sizes.l, A.block_sizes.u)
blockbandwidths(A::BlockSkylineMatrix) = maximum.(colblockbandwidths(A))
BroadcastStyle(::Type{<:BlockSkylineMatrix}) = BlockSkylineStyle()
BroadcastStyle(::Type{<:BlockBandedMatrix}) = BlockBandedStyle()


################################
# AbstractBlockArray Interface #
################################

@inline blocksizes(block_array::BlockSkylineMatrix) = block_array.block_sizes


zeroblock(A::BlockSkylineMatrix, K::Int, J::Int) =
    Matrix(Zeros{eltype(A)}(blocksize(A, (K, J))))

@inline function getblock(A::BlockSkylineMatrix, K::Int, J::Int)
    @boundscheck blockcheckbounds(A, K, J)
    if -A.block_sizes.l[J] ≤ J - K ≤ A.block_sizes.u[J]
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

# @inline function Base.similar(block_array::BlockSkylineMatrix{T}, ::Type{T2}) where {T,N,T2}
#     BlockArray(similar(block_array.blocks, Array{T2, N}), copy(block_array.block_sizes))
# end

Base.size(arr::BlockSkylineMatrix) =
    @inbounds return (cumulsizes(arr.block_sizes.block_sizes,1)[end] - 1, cumulsizes(arr.block_sizes.block_sizes,2)[end] - 1)


@inline function getindex(A::BlockSkylineMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    bi = global2blockindex(A.block_sizes, (i, j))
    @inbounds v = view(A, Block(bi.I))[bi.α...]
    return v
end

@inline function setindex!(A::BlockSkylineMatrix{T}, v, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    bi = global2blockindex(A.block_sizes, (i, j))
    V = view(A, Block(bi.I))
    @inbounds V[bi.α...] = convert(T, v)::T
    return v
end

## structured matrix methods ##
function Base.replace_in_print_matrix(A::BlockSkylineMatrix, i::Integer, j::Integer, s::AbstractString)
    bi = global2blockindex(A.block_sizes, (i, j))
    I,J = bi.I
    -A.block_sizes.l[J] ≤ J-I ≤ A.block_sizes.u[J] ? s : Base.replace_with_centered_mark(s)
end

############
# Indexing #
############

# function _check_setblock!(block_arr::BlockArray{T, N}, v, block::NTuple{N, Int}) where {T,N}
#     for i in 1:N
#         if size(v, i) != blocksize(block_arr.block_sizes, (i, block[i]))
#             throw(DimensionMismatch(string("tried to assign $(size(v)) array to ", blocksize(block_arr, block), " block")))
#         end
#     end
# end
#
#
@inline function setblock!(A::BlockSkylineMatrix, v, K::Int, J::Int)
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
# @generated function Base.copyto!(block_array::BlockArray{T, N, R}, arr::R) where {T,N,R <: AbstractArray}
#     return quote
#         block_sizes = block_array.block_sizes
#
#         @nloops $N i i->(1:nblocks(block_sizes, i)) begin
#             block_index = @ntuple $N i
#             indices = globalrange(block_sizes, block_index)
#             copyto!(getblock(block_array, block_index...), arr[indices...])
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

const BlockBandedBlock{T,LL,UU} = SubArray{T,2,BlockSkylineMatrix{T,LL,UU},Tuple{BlockSlice1,BlockSlice1},false}




# gives the columns of parent(V).data that encode the block
blocks(V::BlockBandedBlock)::Tuple{Int,Int} = first(first(parentindices(V)).block.n),
first(last(parentindices(V)).block.n)

######################################
# Matrix interface  for Blocks #
######################################


MemoryLayout(::Type{<:BlockBandedBlock}) = ColumnMajor()

function Base.unsafe_convert(::Type{Ptr{T}}, V::BlockBandedBlock{T}) where T
    A = parent(V)
    K,J = blocks(V)
    Base.unsafe_convert(Ptr{T}, A.data) + sizeof(T)*(blockstart(A,K,J)-1)
end

strides(V::BlockBandedBlock) = (1,parent(V).block_sizes.block_strides[blocks(V)[2]])

@propagate_inbounds function getindex(V::BlockBandedBlock, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = blocks(V)
    if -A.block_sizes.l[J] ≤ J-K ≤ A.block_sizes.u[J]
        b_start = blockstart(A,K,J)
        b_start == 0 && return zero(eltype(V))
        b_stride = blockstride(A,J)
        A.data[b_start + k-1 + (j-1)*b_stride ]
    else
        zero(eltype(V))
    end
end

@propagate_inbounds function setindex!(V::BlockBandedBlock, v, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = blocks(V)
    if -A.block_sizes.l[J] ≤ J-K ≤ A.block_sizes.u[J]
        b_start = blockstart(A,K,J)
        # TODO: What to do if b_start == 0 ?
        b_stride = A.block_sizes.block_strides[J]
        A.data[b_start + k-1 + (j-1)*b_stride ] = v
    elseif iszero(v) # allow setindex for 0 data
        v
    else
        throw(BandError(A, J-K))
    end
end
