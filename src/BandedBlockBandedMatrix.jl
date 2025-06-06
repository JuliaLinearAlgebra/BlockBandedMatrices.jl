function check_data_sizes(data::AbstractMatrix, raxis, (l,u), (λ,μ))
    if blocksize(data,1) ≠ l + u + 1 && !(blocksize(data,1) == 0 && (-l > u || -λ > μ))
        throw(ArgumentError("Data matrix must have number of row blocks equal to number of block bands"))
    end
    for K = blockaxes(data,1)
        if length(axes(data,1)[K]) ≠ λ + μ + 1 && !(-λ > μ)
            throw(ArgumentError("Data matrix must have row block sizes equal to number of subblock bands"))
        end
    end
end


abstract type AbstractBandedBlockBandedMatrix{T} <: AbstractBlockBandedMatrix{T} end
MemoryLayout(::Type{<:AbstractBandedBlockBandedMatrix}) = BandedBlockBandedLayout()

function _BandedBlockBandedMatrix end


# Represents a block banded matrix with banded blocks
#   similar to BandedMatrix{BandedMatrix{T,Matrix{T}}}
# Here the data is stored by blocks, in a way that is consistent with
# BandedMatrix
#

struct BandedBlockBandedMatrix{T, BLOCKS, RAXIS<:AbstractUnitRange{Int}} <: AbstractBandedBlockBandedMatrix{T}
    data::BLOCKS
    raxis::RAXIS

    l::Int  # block lower bandwidth
    u::Int  # block upper bandwidth
    λ::Int  # sub lower bandwidth
    μ::Int  # sub upper bandwidth

    global function _BandedBlockBandedMatrix(data::AbstractMatrix,
                                             raxis::AbstractUnitRange{Int},
                                             lu::NTuple{2,Int}, λμ::NTuple{2,Int})
      @boundscheck check_data_sizes(data, raxis, lu, λμ)
      new{eltype(data), typeof(data), typeof(raxis)}(data, raxis, lu..., λμ...)
    end
end

const DefaultBandedBlockBandedMatrix{T} = BandedBlockBandedMatrix{T, BlockedMatrix{T, Matrix{T}, NTuple{2,DefaultBlockAxis}}, DefaultBlockAxis}

@inline _BandedBlockBandedMatrix(data::AbstractMatrix, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) =
    _BandedBlockBandedMatrix(BlockedArray(data,(blockedrange(Fill(sum(λμ)+1,sum(lu)+1)),axes[2])), axes[1], lu, λμ)

@inline _BandedBlockBandedMatrix(data::AbstractMatrix,rblocksizes::AbstractVector{Int}, cblocksizes::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) =
    _BandedBlockBandedMatrix(data, (blockedrange(rblocksizes),blockedrange(cblocksizes)), lu, λμ)

_bbb_data_axes(caxes, lu, λμ) = (blockedrange(Fill(max(0,sum(λμ)+1),max(0,sum(lu)+1))),caxes)

BandedBlockBandedMatrix{T,B,R}(::UndefInitializer, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B,R<:AbstractUnitRange{Int}} =
    _BandedBlockBandedMatrix(B(undef, _bbb_data_axes(axes[2],lu,λμ)), axes[1], lu, λμ)
BandedBlockBandedMatrix{T,B,R}(::UndefInitializer, rblocksizes::AbstractVector{Int}, cblocksizes::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B,R<:AbstractUnitRange{Int}} =
    BandedBlockBandedMatrix{T,B,R}(undef, (blockedrange(rblocksizes),blockedrange(cblocksizes)), lu, λμ)

BandedBlockBandedMatrix{T,B}(::UndefInitializer, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B} =
    BandedBlockBandedMatrix{T,B,typeof(axes[1])}(undef,axes,lu,λμ)
BandedBlockBandedMatrix{T,B}(::UndefInitializer, rblocksizes::AbstractVector{Int}, cblocksizes::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B} =
    BandedBlockBandedMatrix{T,B}(undef, (blockedrange(rblocksizes),blockedrange(cblocksizes)), lu, λμ)

BandedBlockBandedMatrix{T}(::UndefInitializer, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where T =
    _BandedBlockBandedMatrix(BlockedMatrix{T}(undef, _bbb_data_axes(axes[2],lu,λμ)), axes[1], lu, λμ)
"""
    BandedBlockBandedMatrix{T}(undef, rows, cols, (l, u), (λ, μ))

Return an unitialized `BandedBlockBandedMatrix` having `eltype` `T`,
with block-bandwidths `(l,u)` and where `A[Block(K,J)]` is a
`BandedMatrix{T}` of size `rows[K]×cols[J]` with bandwidths `(λ,μ)`.
"""
BandedBlockBandedMatrix{T}(::UndefInitializer, rblocksizes::AbstractVector{Int}, cblocksizes::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where T =
    BandedBlockBandedMatrix{T}(undef, (blockedrange(rblocksizes),blockedrange(cblocksizes)), lu, λμ)


"""
    BandedBlockBandedMatrix(M::Union{UniformScaling,AbstractMatrix},
                               rows, cols, (l, u), (λ, μ))

Return a `sum(rows) × sum(cols)` banded-block-banded matrix `A`,
with block-bandwidths `(l,u)` and where `A[Block(K,J)]` is a
`BandedMatrix` of size `rows[K]`×`cols[J]` with bandwidths `(λ,μ)`.
The structural non-zero elements of the returned matrix corresponds to those of `M`.

# Examples

```jldoctest
julia> using LinearAlgebra, FillArrays

julia> BandedBlockBandedMatrix(I, [3,4,3], [3,4,3], (1,1), (1,1))
3×3-blocked 10×10 BandedBlockBandedMatrix{Bool} with block-bandwidths (1, 1) and sub-block-bandwidths block-bandwidths (1, 1):
 1  0  ⋅  │  0  0  ⋅  ⋅  │  ⋅  ⋅  ⋅
 0  1  0  │  0  0  0  ⋅  │  ⋅  ⋅  ⋅
 ⋅  0  1  │  ⋅  0  0  0  │  ⋅  ⋅  ⋅
 ─────────┼──────────────┼─────────
 0  0  ⋅  │  1  0  ⋅  ⋅  │  0  0  ⋅
 0  0  0  │  0  1  0  ⋅  │  0  0  0
 ⋅  0  0  │  ⋅  0  1  0  │  ⋅  0  0
 ⋅  ⋅  0  │  ⋅  ⋅  0  1  │  ⋅  ⋅  0
 ─────────┼──────────────┼─────────
 ⋅  ⋅  ⋅  │  0  0  ⋅  ⋅  │  1  0  ⋅
 ⋅  ⋅  ⋅  │  0  0  0  ⋅  │  0  1  0
 ⋅  ⋅  ⋅  │  ⋅  0  0  0  │  ⋅  0  1

julia> BandedBlockBandedMatrix(Ones{Int}(10,13), [3,4,3], [4,5,4], (1,1), (1,1))
3×3-blocked 10×13 BandedBlockBandedMatrix{Int64} with block-bandwidths (1, 1) and sub-block-bandwidths block-bandwidths (1, 1):
 1  1  ⋅  ⋅  │  1  1  ⋅  ⋅  ⋅  │  ⋅  ⋅  ⋅  ⋅
 1  1  1  ⋅  │  1  1  1  ⋅  ⋅  │  ⋅  ⋅  ⋅  ⋅
 ⋅  1  1  1  │  ⋅  1  1  1  ⋅  │  ⋅  ⋅  ⋅  ⋅
 ────────────┼─────────────────┼────────────
 1  1  ⋅  ⋅  │  1  1  ⋅  ⋅  ⋅  │  1  1  ⋅  ⋅
 1  1  1  ⋅  │  1  1  1  ⋅  ⋅  │  1  1  1  ⋅
 ⋅  1  1  1  │  ⋅  1  1  1  ⋅  │  ⋅  1  1  1
 ⋅  ⋅  1  1  │  ⋅  ⋅  1  1  1  │  ⋅  ⋅  1  1
 ────────────┼─────────────────┼────────────
 ⋅  ⋅  ⋅  ⋅  │  1  1  ⋅  ⋅  ⋅  │  1  1  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  │  1  1  1  ⋅  ⋅  │  1  1  1  ⋅
 ⋅  ⋅  ⋅  ⋅  │  ⋅  1  1  1  ⋅  │  ⋅  1  1  1
```
"""
BandedBlockBandedMatrix


# Auxiliary outer constructors

function convert(::Type{<:BandedBlockBandedMatrix}, B::BandedMatrix)
    if isdiag(B) # TODO really should be one block
        _BandedBlockBandedMatrix(copy(B.data), Fill(1,size(B,1)), Fill(1,size(B,2)), (0,0), (0,0))
    else
        _BandedBlockBandedMatrix(copy(B.data), [size(B,1)], [size(B,2)], (0,0), (B.l,B.u))
    end
end

convert(::Type{BandedBlockBandedMatrix{T,BLOCKS,RAXIS}}, A::BandedBlockBandedMatrix) where {T,BLOCKS,RAXIS} =
    _BandedBlockBandedMatrix(convert(BLOCKS, A.data), convert(RAXIS, A.raxis), (A.l, A.u), (A.λ, A.μ))

function BandedBlockBandedMatrix{T,B,R}(Z::Zeros, axes::NTuple{2,AbstractUnitRange{Int}},
                                       lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B,R<:AbstractUnitRange{Int}}
   if size(Z) ≠ map(length,axes)
       throw(DimensionMismatch())
   end
   blocks = fill!(B(undef, _bbb_data_axes(axes[2],lu,λμ)), zero(T))
   _BandedBlockBandedMatrix(blocks, convert(R, axes[1]), lu, λμ)
end

function BandedBlockBandedMatrix{T,B,R}(E::Eye, axes::NTuple{2,AbstractUnitRange{Int}},
                                       lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B,R<:AbstractUnitRange{Int}}
    if size(E) ≠ map(length,axes)
        throw(DimensionMismatch())
    end
    ret = BandedBlockBandedMatrix{T,B,R}(Zeros{T}(E), axes, lu, λμ)
    ret[diagind(ret)] .= one(T)
    ret
end

function BandedBlockBandedMatrix{T,B,R}(A::UniformScaling, axes::NTuple{2,AbstractUnitRange{Int}},
                                       lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B,R<:AbstractUnitRange{Int}}
    ret = BandedBlockBandedMatrix{T,B,R}(Zeros{T}(map(length,axes)), axes, lu, λμ)
    ret[diagind(ret)] .= convert(T, A.λ)
    ret
end

BandedBlockBandedMatrix{T,B}(m::Union{AbstractMatrix, UniformScaling},
                            axes::NTuple{2,AbstractUnitRange{Int}},
                           lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B} =
  BandedBlockBandedMatrix{T,B,typeof(axes[1])}(m, axes, lu, λμ)

BandedBlockBandedMatrix{T}(m::Union{AbstractMatrix, UniformScaling},
                            axes::NTuple{2,AbstractUnitRange{Int}},
                           lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where T =
  DefaultBandedBlockBandedMatrix{T}(m, axes, lu, λμ)

BandedBlockBandedMatrix{T,B,R}(m::Union{AbstractMatrix, UniformScaling},
                            rdims::AbstractVector{Int}, cdims::AbstractVector{Int},
                           lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B,R<:AbstractUnitRange{Int}} =
  BandedBlockBandedMatrix{T,B,R}(m, (blockedrange(rdims),blockedrange(cdims)), lu, λμ)

BandedBlockBandedMatrix{T,B}(m::Union{AbstractMatrix, UniformScaling},
                            rdims::AbstractVector{Int}, cdims::AbstractVector{Int},
                           lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B} =
  BandedBlockBandedMatrix{T,B}(m, (blockedrange(rdims),blockedrange(cdims)), lu, λμ)

BandedBlockBandedMatrix{T}(m::Union{AbstractMatrix, UniformScaling},
                            rdims::AbstractVector{Int}, cdims::AbstractVector{Int},
                           lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where T =
  BandedBlockBandedMatrix{T}(m, (blockedrange(rdims),blockedrange(cdims)), lu, λμ)


BandedBlockBandedMatrix(A::Union{AbstractMatrix,UniformScaling},
                        axes::NTuple{2, AbstractUnitRange{Int}}, lu::NTuple{2,Int},
                        λμ::NTuple{2,Int}) =
    BandedBlockBandedMatrix{eltype(A)}(A, axes, lu, λμ)
BandedBlockBandedMatrix(A::Union{AbstractMatrix,UniformScaling},
                        rdims::AbstractVector{Int}, cdims::AbstractVector{Int},
                        lu::NTuple{2,Int}, λμ::NTuple{2,Int}) =
    BandedBlockBandedMatrix(A, (blockedrange(rdims),blockedrange(cdims)), lu, λμ)


function BandedBlockBandedMatrix{T,Blocks,RR}(A::AbstractMatrix, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,Blocks,RR<:AbstractUnitRange{Int}}
    ret = BandedBlockBandedMatrix{T,Blocks,RR}(Zeros{T}(size(A)), axes, lu, λμ)
    L,M = λμ
    for J = blockaxes(ret,2), K = blockcolsupport(ret, J)
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

BandedBlockBandedMatrix{T}(A::AbstractMatrix, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where T =
    copyto!(BandedBlockBandedMatrix{T}(undef, axes(A), lu, λμ), A)
BandedBlockBandedMatrix{T}(A::AbstractMatrix) where T = BandedBlockBandedMatrix{T}(A, blockbandwidths(A), subblockbandwidths(A))

BandedBlockBandedMatrix(A::AbstractMatrix{T}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where T = BandedBlockBandedMatrix{T}(A, lu, λμ)
BandedBlockBandedMatrix(A::AbstractMatrix{T}) where T = BandedBlockBandedMatrix{T}(A)

BandedBlockBandedMatrix(A::AbstractMatrix, rdims::AbstractVector{Int}, cdims::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) =
    BandedBlockBandedMatrix(A, (blockedrange(rdims), blockedrange(cdims)), lu, λμ)

copy(B::BandedBlockBandedMatrix) = _BandedBlockBandedMatrix(copy(B.data), B.raxis, blockbandwidths(B), subblockbandwidths(B))
AbstractArray{T}(B::BandedBlockBandedMatrix) where T = _BandedBlockBandedMatrix(AbstractArray{T}(B.data), B.raxis, blockbandwidths(B), subblockbandwidths(B))
AbstractMatrix{T}(B::BandedBlockBandedMatrix) where T = _BandedBlockBandedMatrix(AbstractMatrix{T}(B.data), B.raxis, blockbandwidths(B), subblockbandwidths(B))
convert(::Type{AbstractArray{T}}, B::BandedBlockBandedMatrix) where T = _BandedBlockBandedMatrix(convert(AbstractArray{T},B.data), B.raxis, blockbandwidths(B), subblockbandwidths(B))
convert(::Type{AbstractMatrix{T}}, B::BandedBlockBandedMatrix) where T = _BandedBlockBandedMatrix(convert(AbstractMatrix{T},B.data), B.raxis, blockbandwidths(B), subblockbandwidths(B))
convert(::Type{AbstractArray{T}}, B::BandedBlockBandedMatrix{T}) where T = B
convert(::Type{AbstractMatrix{T}}, B::BandedBlockBandedMatrix{T}) where T = B


similar(A::BandedBlockBandedMatrix, ::Type{T}, axes::NTuple{2,AbstractUnitRange{Int}}) where T =
    BandedBlockBandedMatrix{T}(undef, axes, blockbandwidths(A), subblockbandwidths(A))

@inline similar(A::BandedBlockBandedMatrix, ::Type{T}, axes::Tuple{AbstractBlockedUnitRange,AbstractUnitRange{Int}}) where T =
    BandedBlockBandedMatrix{T}(undef, axes, blockbandwidths(A), subblockbandwidths(A))
@inline similar(A::BandedBlockBandedMatrix, ::Type{T}, axes::Tuple{AbstractUnitRange{Int},AbstractBlockedUnitRange}) where T =
    BandedBlockBandedMatrix{T}(undef, axes, blockbandwidths(A), subblockbandwidths(A))
@inline similar(A::BandedBlockBandedMatrix, ::Type{T}, axes::Tuple{AbstractBlockedUnitRange,AbstractBlockedUnitRange}) where T =
    BandedBlockBandedMatrix{T}(undef, axes, blockbandwidths(A), subblockbandwidths(A))


similar(A::BandedBlockBandedMatrix{T}, axes::NTuple{2,AbstractUnitRange{Int}}) where T =
    similar(Matrix{T}, map(length,axes)...)

axes(A::BandedBlockBandedMatrix) = (A.raxis, axes(A.data,2))

bandedblockbandeddata(A::BandedBlockBandedMatrix) = A.data

################################
# BandedBlockBandedMatrix Interface #
################################

bandedblockbandedcolumns(L::AbstractColumnMajor) = BandedBlockBandedColumnMajor()
bandedblockbandedcolumns(L::AbstractRowMajor) = BandedBlockBandedColumns{RowMajor}()
bandedblockbandedcolumns(::DualLayout{ML}) where ML = bandedblockbandedcolumns(ML())
bandedblockbandedcolumns(_) = BandedBlockBandedColumns{UnknownLayout}()

MemoryLayout(::Type{<:BandedBlockBandedMatrix{<:Any,BLOCKS}}) where BLOCKS =
    bandedblockbandedcolumns(MemoryLayout(BLOCKS))
bandedblockbandedbroadcaststyle(_) = BandedBlockBandedStyle()
BroadcastStyle(::Type{<:BandedBlockBandedMatrix{<:Any,BLOCKS}}) where BLOCKS =
    bandedblockbandedbroadcaststyle(BroadcastStyle(BLOCKS))

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

# default is to use whole block
_subblockbandwidths(A::AbstractMatrix, ::NTuple{2,OneTo{Int}}) = bandwidths(A)
function _subblockbandwidths(A::AbstractMatrix, _)
    M,N = map(x -> maximum(x, init=0), blocklengths.(axes(A)))
    M-1,N-1
end

subblockbandwidths(A::AbstractMatrix) = _subblockbandwidths(A, axes(A))

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


zeroblock(A::BandedBlockBandedMatrix, K::Int, J::Int) =
    BandedMatrix(Zeros{eltype(A)}(length.(getindex.(axes(A),(Block(K),Block(J))))), (A.λ, A.μ))

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
    @inbounds return map(length,axes(arr))


@inline function getindex(A::BandedBlockBandedMatrix{T}, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    bi = findblockindex.(axes(A), (i,j))
    @inbounds v = view(A, block.(bi)...)[blockindex.(bi)...]
    return v
end

@inline function setindex!(A::BandedBlockBandedMatrix{T}, v, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    BI,BJ = findblockindex.(axes(A), (i,j))
    if -A.l ≤ Int(block(BJ)-block(BI)) ≤ A.u
        V = view(A, block(BI),block(BJ))
         @inbounds V[blockindex(BI),blockindex(BJ)] = convert(T, v)::T
    elseif !iszero(v)
        throw(BandError(A))
    end
    return v
end

## structured matrix methods ##
function layout_replace_in_print_matrix(::AbstractBandedBlockBandedLayout, A, i, j, s)
    bi = findblockindex.(axes(A), (i,j))
    I,J = block.(bi)
    i,j = blockindex.(bi)
    l,u = blockbandwidths(A)
    λ,μ = subblockbandwidths(A)
    -l ≤ Int(J-I) ≤ u && -λ ≤ j-i ≤ μ ? s : Base.replace_with_centered_mark(s)
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



##################
# BandedBlockBandedBlock
#
#   views of the blocks satisfy the BandedMatrix interface, and are memory-compatible
#   with BLASBandedMatrix.
##################

const SubBandedBlockBandedMatrix{T,R1,R2} =
    SubArray{T,2,<:BandedBlockBandedMatrix{T},<:Tuple{BlockSlice{R1},BlockSlice{R2}}}

const SingleBlockInd = Union{BlockSlice1, BlockSlice{<:BlockIndexRange1}}
const BlockRangeInd = Union{BlockSlices, AbstractBlockedUnitRange}

sublayout(::BandedBlockBandedColumns{ML}, ::Type{II}) where {ML,II<:Tuple{SingleBlockInd,SingleBlockInd}} = bandedcolumns(sublayout(ML(), II))
sublayout(::BandedBlockBandedColumns{ML}, ::Type{II}) where {ML,II<:Tuple{BlockRangeInd,BlockRangeInd}} = bandedblockbandedcolumns(sublayout(ML(), II))
sublayout(::BandedBlockBandedColumns{ML}, ::Type{II}) where {ML,II<:Tuple{SingleBlockInd,BlockRangeInd}} = bandedblockbandedcolumns(sublayout(ML(), II))
sublayout(::BandedBlockBandedColumns{ML}, ::Type{II}) where {ML,II<:Tuple{BlockRangeInd,SingleBlockInd}} = bandedblockbandedcolumns(sublayout(ML(), II))

_blockaxes1(A::BlockSlice) = A.block
_blockaxes1(A::Slice) = _blockaxes1(A.indices)
_blockaxes1(A::AbstractBlockedUnitRange) = blockaxes(A,1)
blockbandshift(A, B) = BandedMatrices.bandshift(Int.(_blockaxes1(A)), Int.(_blockaxes1(B)))
blockbandshift(S) = blockbandshift(parentindices(S)...)

function bandedblockbandeddata(V::SubArray)
    l,u = blockbandwidths(V)
    L,U = blockbandwidths(parent(V)) .+ (-1,1) .* blockbandshift(V)
    view(bandedblockbandeddata(parent(V)), Block.(U-u+1:U+l+1), parentindices(V)[2])
end

sublayout(::AbstractBandedBlockBandedLayout, ::Type{<:Tuple{SingleBlockInd,SingleBlockInd}}) = BandedLayout()
sublayout(::AbstractBandedBlockBandedLayout, ::Type{<:Tuple{BlockRangeInd,BlockRangeInd}}) = BandedBlockBandedLayout()
sublayout(::AbstractBandedBlockBandedLayout, ::Type{<:Tuple{SingleBlockInd,BlockRangeInd}}) = BandedBlockBandedLayout()
sublayout(::AbstractBandedBlockBandedLayout, ::Type{<:Tuple{BlockRangeInd,SingleBlockInd}}) = BandedBlockBandedLayout()


sub_materialize(::AbstractBandedBlockBandedLayout, V, _) = BandedBlockBandedMatrix(V)
sub_materialize(::AbstractBandedBlockBandedLayout, V, ::Tuple{AbstractBlockedUnitRange,AbstractBlockedUnitRange}) = BandedBlockBandedMatrix(V)
sub_materialize(::AbstractBandedBlockBandedLayout, V, ::Tuple{AbstractUnitRange,AbstractBlockedUnitRange}) = BlockedArray(V)
sub_materialize(::AbstractBandedBlockBandedLayout, V, ::Tuple{AbstractBlockedUnitRange,AbstractUnitRange}) = BlockedArray(V)



isbanded(A::SubArray{<:Any,2,<:BandedBlockBandedMatrix}) = MemoryLayout(A) isa AbstractBandedLayout
isbandedblockbanded(A::SubArray{<:Any,2,<:BandedBlockBandedMatrix}) = MemoryLayout(A) isa AbstractBandedBlockBandedLayout


subblockbandwidths(V::SubArray) = subblockbandwidths(parent(V))


_firstblock(B::AbstractBlockedUnitRange) = Int(first(blockaxes(B,1)))
_firstblock(B::Block{1}) = Int(B)
_firstblock(B::BlockIndexRange) = _firstblock(B.block)
_firstblock(B::BlockRange) = first(B.indices[1])
_firstblock(B::Base.Slice) = _firstblock(B.indices)
_firstblock(B::BlockSlice) = _firstblock(B.block)


function blockbandwidths(V::SubArray)
    A = parent(V)
    K1,J1 = map(_firstblock, parentindices(V))
    shift = K1-J1
    l,u = blockbandwidths(A)
    l - shift, u + shift
end


const BandedBlockBandedBlock{T, BLOCKS, RAXIS} = SubArray{T,2,BandedBlockBandedMatrix{T, BLOCKS, RAXIS},<:Tuple{BlockSlice1,BlockSlice1},false}


BroadcastStyle(::Type{<: BandedBlockBandedBlock}) = BandedStyle()


function parentblock(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice{<:BlockIndexRange1}}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(A, K_sl.block.block, J_sl.block.block)
end

function parentblock(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice1,BlockSlice{<:BlockIndexRange1}}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(A, K_sl.block, J_sl.block.block)
end

function parentblock(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice1}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(A, K_sl.block.block, J_sl.block)
end

# gives the columns of parent(V).data that encode the block
parentblocks2Int(V::BandedBlockBandedBlock)::Tuple{Int,Int} = Int(first(parentindices(V)).block),
                                                          Int(last(parentindices(V)).block)


######################################
# BandedMatrix interface  for Blocks #
######################################
getindex(A::BandedBlockBandedMatrix, KJ::Block{2}) = A[Block.(KJ.n)...]

function inblockbands(A, K::Block{1}, J::Block{1})
    l,u = blockbandwidths(A)
    -l ≤ Int(J-K) ≤ u
end

function inblockbands(V::SubArray{<:Any,2,<:AbstractMatrix,<:Tuple{BlockSlice1,BlockSlice1},false})
    K_sl, J_sl = parentindices(V)
    inblockbands(parent(V), K_sl.block, J_sl.block)
end

function getindex(A::BandedBlockBandedMatrix{T}, K::Block{1}, J::Block{1}) where T
    inblockbands(A, K, J) || return BandedMatrix{T}(undef, (length(axes(A,1)[K]), length(axes(A,2)[J])), (-720,-720))
    _BandedMatrix(A.data[A.u + K - J + 1, J], length(axes(A,1)[K]), subblockbandwidths(A)...)
end

function getindex(A::BandedBlockBandedMatrix{T}, KR::BlockRange{1}, JR::BlockRange{1}) where T
    sh = Int(KR[1]-JR[1])
    l,u = blockbandwidths(A)
    _BandedBlockBandedMatrix(A.data[:,JR], axes(axes(A,1)[KR],1), (l-sh, u+sh), subblockbandwidths(A))
end

@inline function bandwidths(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice1,BlockSlice1}}) where T
    inblockbands(V) && return subblockbandwidths(parent(V))
    (-720,-720)
end

function bandwidths(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice{<:BlockIndexRange1}}}) where T
   B = parentblock(V)
   K_sl, J_sl = parentindices(V)
   bandwidths(B) .+ (-1,1) .* bandshift(K_sl.block.indices[1],J_sl.block.indices[1])
end

function bandwidths(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice1,BlockSlice{<:BlockIndexRange1}}}) where T
    B = parentblock(V)
    K_sl, J_sl = parentindices(V)
    bandwidths(B) .+ (-1,1) .* bandshift(Base.OneTo(1),J_sl.block.indices[1])
 end

function bandwidths(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice1}}) where T
    B = parentblock(V)
    K_sl, J_sl = parentindices(V)
    bandwidths(B) .+ (-1,1) .* bandshift(K_sl.block.indices[1],Base.OneTo(1))
 end


function bandeddata(V::BandedBlockBandedBlock{T}) where T
    inblockbands(V) || return Array{T}(undef, 0, size(V,2))
    A = parent(V)
    u = A.u
    K_sl, J_sl = parentindices(V)
    K, J = K_sl.block, J_sl.block
    view(A.data, u + K - J + 1, J)
end

function bandeddata(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice{<:BlockIndexRange1}}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(bandeddata(parentblock(V)), :, J_sl.block.indices[1])
end

function bandeddata(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice1,BlockSlice{<:BlockIndexRange1}}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(bandeddata(parentblock(V)), :, J_sl.block.indices[1])
end

function bandeddata(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice1}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    bandeddata(parentblock(V))
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
    K,J = parentblocks2Int(V)
    if -A.l ≤ J-K ≤ A.u && -A.λ ≤ j-k ≤ A.μ
        inbands_getindex(V, k, j)
    else
        zero(eltype(V))
    end
end

@propagate_inbounds function setindex!(V::BandedBlockBandedBlock, v, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = parentblocks2Int(V)
    if -A.l ≤ J-K ≤ A.u && -A.λ ≤ j-k ≤ A.μ
        inbands_setindex!(V, v, k, j)
    elseif iszero(v) # allow setindex for 0 datya
        v
    else
        throw(BandError(V, J-K))
    end
end


axpy!(a::T, A::BandedBlockBandedMatrix{T}, B::BandedBlockBandedMatrix{T}) where {T} =
    B .= a .* A .+ B


lmul!(x::Number, A::BandedBlockBandedMatrix) = (lmul!(x, A.data); A)
rmul!(A::BandedBlockBandedMatrix, x::Number) = (rmul!(A.data, x); A)
*(x::Number, A::BandedBlockBandedMatrix) = _BandedBlockBandedMatrix(x*A.data, axes(A,1), blockbandwidths(A), subblockbandwidths(A))
*(A::BandedBlockBandedMatrix, x::Number) = _BandedBlockBandedMatrix(A.data*x, axes(A,1), blockbandwidths(A), subblockbandwidths(A))


#####
# summary
#####

_bandedblockbanded_summary(io, B::BandedBlockBandedMatrix{T}) where T = print(io, "BandedBlockBandedMatrix{$T} with block-bandwidths $(blockbandwidths(B)) and sub-block-bandwidths block-bandwidths $(subblockbandwidths(B))")
BlockArrays._show_typeof(io::IO, B::DefaultBandedBlockBandedMatrix) = _bandedblockbanded_summary(io, B)
function BlockArrays._show_typeof(io::IO, B::BandedBlockBandedMatrix)
    _bandedblockbanded_summary(io, B)
    print(io, " with data ")
    summary(io, B.data)
end
