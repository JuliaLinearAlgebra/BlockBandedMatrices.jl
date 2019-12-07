

function check_data_sizes(data::AbstractBlockMatrix, raxis, (l,u), (λ,μ))
    if blocksize(raxis,1) ≠ l + u + 1 && !(blocksize(raxis,1) == 0 && -l > u)
        throw(ArgumentError("Data matrix must have number of row blocks equal to number of block bands"))
    end
    for K = blockaxes(raxis,1)
        if length(raxis[K]) ≠ λ + μ + 1 && !(-λ > μ)
            throw(ArgumentError("Data matrix must have row block sizes equal to number of subblock bands"))
        end
    end
end


function _BandedBlockBandedMatrix end


# Represents a block banded matrix with banded blocks
#   similar to BandedMatrix{BandedMatrix{T,Matrix{T}}}
# Here the data is stored by blocks, in a way that is consistent with
# BandedMatrix
#

struct BandedBlockBandedMatrix{T, BLOCKS, RAXIS<:AbstractUnitRange{Int}} <: AbstractBlockBandedMatrix{T}
    data::BLOCKS
    raxes::RAXIS

    l::Int  # block lower bandwidth
    u::Int  # block upper bandwidth
    λ::Int  # sub lower bandwidth
    μ::Int  # sub upper bandwidth

    global function _BandedBlockBandedMatrix(data::AbstractBlockMatrix,
                                             raxis::AbstractUnitRange{Int},
                                             lu::NTuple{2,Int}, λμ::NTuple{2,Int})
      @boundscheck check_data_sizes(data, raxis, lu, λμ)
      new{eltype(data), typeof(data)}(data, raxis, lu..., λμ...)
    end
end

const DefaultBandedBlockBandedMatrix{T} = BandedBlockBandedMatrix{T, <:PseudoBlockMatrix{T, Matrix{T}, <:Any}}

_BandedBlockBandedMatrix(data::AbstractBlockMatrix,rblocksizes::AbstractVector{Int}, cblocksizes::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) =
    _BandedBlockBandedMatrix(data, (CumsumBlockRange(rblocksizes),CumsumBlockRange(cblocksizes)), lu, λμ)
                                            

@inline _BandedBlockBandedMatrix(data::AbstractMatrix, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) = 
    _BandedBlockBandedMatrix(PseudoBlockArray(data),  raxis, lu, λμ)

_bbb_data_axes(caxes, lu, λμ) = (CumsumBlockRange(Fill(sum(λμ)+1,sum(lu)+1)),caxes)

BandedBlockBandedMatrix{T, B}(::UndefInitializer, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T, B} =
    _BandedBlockBandedMatrix(B(undef, _bbb_data_axes(axes[2],lu,λμ)), axes[1], lu, λμ)

BandedBlockBandedMatrix{T, B}(::UndefInitializer, rblocksizes::AbstractVector{Int}, cblocksizes::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T, B} =
    BandedBlockBandedMatrix{T, B}(undef, (CumsumBlockRange(rblocksizes),CumsumBlockRange(cblocksizes)), lu, λμ)


"""
    BandedBlockBandedMatrix{T}(undef, (rows, cols), (l, u), (λ, μ))

returns an undef `sum(rows)`×`sum(cols)` banded-block-banded matrix `A`
of type `T` with block-bandwidths `(l,u)` and where `A[Block(K,J)]`
is a `BandedMatrix{T}` of size `rows[K]`×`cols[J]` with bandwidths `(λ,μ)`.
"""
BandedBlockBandedMatrix


BandedBlockBandedMatrix{T}(::UndefInitializer, args...) where T =
    DefaultBandedBlockBandedMatrix{T}(undef, args...)

# Auxiliary outer constructors

function convert(::Type{<:BandedBlockBandedMatrix}, B::BandedMatrix)
    if isdiag(B) # TODO really should be one block
        _BandedBlockBandedMatrix(copy(B.data), Fill(1,size(B,1)), Fill(1,size(B,2)), (0,0), (0,0))
    else
        _BandedBlockBandedMatrix(copy(B.data), [size(B,1)], [size(B,2)], (0,0), (B.l,B.u))
    end
end

BandedBlockBandedMatrix{T, B}(Z::Zeros, axes::NTuple{2,AbstractUnitRange{Int}},
                                lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T, B}
   if size(Z) ≠ map(length,axes)
       throw(DimensionMismatch())
   end
   blocks = fill!(B(undef, _bbb_data_axes(axes[2],lu,λμ)), zero(T))
   _BandedBlockBandedMatrix(blocks, bs)
end

function BandedBlockBandedMatrix{T, B}(E::Eye, axes::NTuple{2,AbstractUnitRange{Int}},
                                       lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T, B}
    if size(Z) ≠ map(length,axes)
        throw(DimensionMismatch())
    end
    ret = BandedBlockBandedMatrix{T, B}(Zeros{T}(E), axes, lu, λμ)
    ret[diagind(ret)] .= one(T)
    ret
end

function BandedBlockBandedMatrix{T, B}(A::UniformScaling, axes::NTuple{2,AbstractUnitRange{Int}},
                                       lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T, B}
    ret = BandedBlockBandedMatrix{T, B}(Zeros{T}(map(length,axes)), axes, lu, λμ)
    ret[diagind(ret)] .= convert(T, A.λ)
    ret
end

BandedBlockBandedMatrix{T}(m::Union{Zeros, Eye, UniformScaling},
                            axes::NTuple{2,AbstractUnitRange{Int}},
                           lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where T =
  DefaultBandedBlockBandedMatrix{T}(m, axes, lu, λμ)


function BandedBlockBandedMatrix{T, Blocks}(A::AbstractMatrix, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T, Blocks}
    ret = BandedBlockBandedMatrix{T, Blocks}(Zeros{T}(size(A)), axes)
    L,M = λμ 
    for J = Block.(1:nblocks(ret, 2)), K = blockcolrange(ret, Int(J))
        kr, jr = axes[1][K], axes[2][J]

        # We have the correct block - now we need to only add the entries from
        # the correct bands
        B = view(A, kr, jr)
        R = view(ret, K, J)
        rows, cols = size(B)
        for λ = 1:L, j = 1:min(rows-λ, cols)
            view(R, j+λ, j) .= view(B, j+λ, j)
        end
        for i = 1:min(rows, cols)
            view(R, i, i) .= view(B, i, i)
        end
        for μ = 1:M, k = 1:min(rows, cols-μ)
            view(R, k, k+μ) .= view(B, k, k+μ)
        end
    end
    ret
end


BandedBlockBandedMatrix(A::AbstractMatrix) =
    BandedBlockBandedMatrix(A, axes(A), blockbandwidths(A), subblockbandwidths(A))

similar(A::BandedBlockBandedMatrix, T::Type=eltype(A), axes::NTuple{2,AbstractUnitRange{Int}}) =
      BandedBlockBandedMatrix{T}(undef, axes)


function sparse(A::BandedBlockBandedMatrix)
    i = Vector{Int}()
    j = Vector{Int}()
    z = Vector{eltype(A)}()
    for J = Block.(1:nblocks(A,2)), K = blockcolrange(A, J)
        B = view(A, K, J)
        ĩ = _banded_rowval(B)
        j̃ = _banded_colval(B)
        z̃ = _banded_nzval(B)
        ĩ .+= cumulsizes(A, 1, Int(K))-1
        j̃ .+= cumulsizes(A, 2, Int(J))-1
        append!(i, ĩ)
        append!(j, j̃)
        append!(z, z̃)
    end
    sparse(i, j, z, size(A)...)
end

################################
# BandedBlockBandedMatrix Interface #
################################

MemoryLayout(::Type{<:BandedBlockBandedMatrix}) = BandedBlockBandedColumnMajor()
BroadcastStyle(::Type{<:BandedBlockBandedMatrix}) = BandedBlockBandedStyle()

isbandedblockbanded(_) = false
isbandedblockbanded(::BandedBlockBandedMatrix) = true

blockbandwidths(A::BandedBlockBandedMatrix) = (A.l, A.u)

"""
    subblockbandwidths(A)

returns the sub-block bandwidths of `A`, where `A` is a banded-block-banded
matrix. In other words, `A[Block(K,J)]` will return a `BandedMatrix` with
bandwidths given by `subblockbandwidths(A)`.
"""
subblockbandwidths(A::BandedBlockBandedMatrix) = (A.λ, A.μ)

# default is to use bandwidth
subblockbandwidths(A::AbstractMatrix) = bandwidths(A)

"""
    subblockbandwidth(A, i)

returns the sub-block lower (`i == 1`) or upper (`i == 2`) bandwidth of `A`,
where `A` is a banded-block-banded matrix. In other words, `A[Block(K,J)]` will
return a `BandedMatrix` with the returned lower/upper bandwidth.
"""
subblockbandwidth(A::AbstractArray, k::Integer) = subblockbandwidths(A)[k]

isdiag(A::BandedBlockBandedMatrix) = A.λ == A.μ == A.l == A.u


################################
# AbstractBlockArray Interface #
################################

@inline blocksizes(block_array::BandedBlockBandedMatrix) = block_array.block_sizes

zeroblock(A::BandedBlockBandedMatrix, K::Int, J::Int) =
    BandedMatrix(Zeros{eltype(A)}(blocksize(A, (K, J))), (A.λ, A.μ))

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
    @inbounds return size(arr.block_sizes)


@inline function getindex(A::BandedBlockBandedMatrix{T}, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    bi = global2blockindex(A.block_sizes, (i, j))
    @inbounds v = view(A, Block(bi.I))[bi.α...]
    return v
end

@inline function setindex!(A::BandedBlockBandedMatrix{T}, v, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    bi = global2blockindex(A.block_sizes, (i, j))
    if -A.l ≤ bi.I[2] - bi.I[1] ≤ A.u
        V = view(A, Block(bi.I))
        @inbounds V[bi.α...] = convert(T, v)::T
    elseif !iszero(v)
        throw(BandError(A))
    end
    return v
end

## structured matrix methods ##
function Base.replace_in_print_matrix(A::BandedBlockBandedMatrix, i::Integer, j::Integer, s::AbstractString)
    bi = global2blockindex(A.block_sizes, (i, j))
    I,J = bi.I
    i,j = bi.α
    -A.l ≤ J-I ≤ A.u && -A.λ ≤ j-i ≤ A.μ ? s : Base.replace_with_centered_mark(s)
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
@inline function setblock!(A::BandedBlockBandedMatrix, v, K::Int, J::Int)
    @boundscheck blockcheckbounds(A, K, J)

    @boundscheck (bandwidth(v, 1) > A.λ || bandwidth(v, 2) > A.μ) && throw(BandError())
    V = view(A, Block(K), Block(J))
    V .= v
    return A
end
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
# BandedBlockBandedBlock
#
#   views of the blocks satisfy the BandedMatrix interface, and are memory-compatible
#   with BLASBandedMatrix.
##################

const BandedBlockBandedBlock{T, BLOCKS} = SubArray{T,2,BandedBlockBandedMatrix{T, BLOCKS},Tuple{BlockSlice1,BlockSlice1},false}


isbanded(::BandedBlockBandedBlock) = true
MemoryLayout(::BandedBlockBandedBlock) = BandedColumnMajor()
BroadcastStyle(::Type{<: BandedBlockBandedBlock}) = BandedStyle()


function inblockbands(V::BandedBlockBandedBlock)
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    K, J = K_sl.block, J_sl.block
    -A.l ≤ Int(J-K) ≤ A.u
end


######################################
# BandedMatrix interface  for Blocks #
######################################
@inline function bandwidths(V::BandedBlockBandedBlock) 
    inblockbands(V) && return subblockbandwidths(parent(V))
    (-720,-720)
end



# gives the columns of parent(V).data that encode the block
blocks(V::BandedBlockBandedBlock)::Tuple{Int,Int} = Int(first(parentindices(V)).block),
                                                    Int(last(parentindices(V)).block)


function bandeddata(V::BandedBlockBandedBlock{T}) where T
    inblockbands(V) || return Array{T}(undef, 0, size(V,2))
    A = parent(V)
    u = A.u
    K_sl, J_sl = parentindices(V)
    K, J = K_sl.block, J_sl.block
    view(A.data, u + K - J + 1, J)
end


@inline function inbands_getindex(V::BandedBlockBandedBlock, k::Int, j::Int)
    A = parent(V)
    banded_getindex(bandeddata(V), A.λ, A.μ, k, j)
end

@inline function inbands_setindex!(V::BandedBlockBandedBlock, v, k::Int, j::Int)
    A = parent(V)
    banded_setindex!(bandeddata(V), A.λ, A.μ, v, k, j)
end

# we need getindex as we use block-wise get index above
@propagate_inbounds function getindex(V::BandedBlockBandedBlock, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = blocks(V)
    if -A.l ≤ J-K ≤ A.u && -A.λ ≤ j-k ≤ A.μ
        inbands_getindex(V, k, j)
    else
        zero(eltype(V))
    end
end

@propagate_inbounds function setindex!(V::BandedBlockBandedBlock, v, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = blocks(V)
    if -A.l ≤ J-K ≤ A.u && -A.λ ≤ j-k ≤ A.μ
        inbands_setindex!(V, v, k, j)
    elseif iszero(v) # allow setindex for 0 datya
        v
    else
        throw(BandError(V, J-K))
    end
end


BLAS.axpy!(a::T, A::BandedBlockBandedMatrix{T}, B::BandedBlockBandedMatrix{T}) where T =
    B .= a .* A .+ B


lmul!(x::Number, A::BandedBlockBandedMatrix) = (lmul!(x, A.data); A)
rmul!(A::BandedBlockBandedMatrix, x::Number) = (rmul!(A.data, x); A)
*(x::Number, A::BandedBlockBandedMatrix) = _BandedBlockBandedMatrix(x*A.data, A.block_sizes)
*(A::BandedBlockBandedMatrix, x::Number) = _BandedBlockBandedMatrix(A.data*x, A.block_sizes)
