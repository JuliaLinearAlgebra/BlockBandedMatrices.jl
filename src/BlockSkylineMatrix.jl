checkbandwidths(N, M, l::AbstractVector{Int}, u::AbstractVector{Int}) =
    M == 1 || (length(u) == M && length(l) == M) ||
    throw(DimensionMismatch("For a matrix of $(N)×$(M) blocks, $(M) lower and upper column bandwidths are required"))

#### Routines for BandedSizes
function bb_blockstarts(ax, l::AbstractVector{Int}, u::AbstractVector{Int})
    N,M = blocksize.(ax,1)
    L,U = maximum(l), maximum(u)
    b_start = BandedMatrix{Int}(undef, (N, M), (L, U))
    -L > U && return b_start

    checkbandwidths(N, M, l, u)

    ind_shift = 0
    for J = 1:M
        KR = Block.(max(1,J-u[J]):min(J+l[J],N))
        if !isempty(KR)
            b_start[Int.(KR),J] .= ind_shift .+ first.(getindex.(Ref(ax[1]),KR)) .- first(ax[1][KR[1]]) .+ 1

            num_rows = length(ax[1][KR])
            num_cols = length(ax[2][Block(J)])
            ind_shift += num_rows*num_cols
        end
    end

    b_start
end

function bb_blockstrides(b_axes, l::AbstractVector{Int}, u::AbstractVector{Int})
    N, M = blocksize.(b_axes,1)
    L,U = maximum(l), maximum(u)
    checkbandwidths(N, M, l, u)
    b_strides = Vector{Int}(undef, M)
    for J=1:M
        KR = Block.(max(1,J-u[J]):min(J+l[J],N))
        if !isempty(KR)
            b_strides[J] = length(b_axes[1][KR])
        else
            b_strides[J] = 0
        end
    end
    b_strides
end

bb_blockstrides(b_axes, l::Integer, u::Integer) = bb_blockstrides(b_axes, Fill(l, blocksize(b_axes,1)), Fill(u, blocksize(b_axes,2)))

struct BlockSkylineSizes{BS<:NTuple{2,AbstractUnitRange{Int}}, LL<:AbstractVector{Int}, UU<:AbstractVector{Int}, BStarts, BStrides}
    axes::BS
    block_starts::BStarts # gives index where the blocks start, usually a BandedMatrix{Int}
    block_strides::BStrides # gives stride to next block for J-th column, usually a Vector{Int}
    l::LL
    u::UU
end

const BlockBandedSizes = BlockSkylineSizes{NTuple{2,BlockedUnitRange{Vector{Int}}}, Fill{Int,1,Tuple{OneTo{Int}}}, Fill{Int,1,Tuple{OneTo{Int}}},
                                            BandedMatrix{Int,Matrix{Int},OneTo{Int}}, Vector{Int}}


BlockSkylineSizes(b_axes::NTuple{2,AbstractUnitRange{Int}}, l::AbstractVector{Int}, u::AbstractVector{Int}) =
    BlockSkylineSizes(b_axes, bb_blockstarts(b_axes, l, u), bb_blockstrides(b_axes, l, u), l, u)

BlockSkylineSizes(rows::AbstractVector{Int}, cols::AbstractVector{Int}, l::AbstractVector{Int}, u::AbstractVector{Int}) =
    BlockSkylineSizes((blockedrange(rows),blockedrange(cols)), l, u)

BlockBandedSizes(b_axes::NTuple{2,AbstractUnitRange{Int}}, l::Int, u::Int) =
    convert(BlockBandedSizes, BlockSkylineSizes(b_axes, Fill(l, blocklength(b_axes[2])), Fill(u, blocklength(b_axes[2]))))


BlockBandedSizes(rows::AbstractVector{Int}, cols::AbstractVector{Int}, l::Int, u::Int) =
    convert(BlockBandedSizes, BlockSkylineSizes(rows, cols, Fill(l, length(cols)), Fill(u, length(cols))))

convert(::Type{BlockSkylineSizes{BS,LL,UU,BStarts,BStrides}}, b::BlockSkylineSizes) where {BS,LL,UU,BStarts,BStrides} =
    BlockSkylineSizes(convert(BS, b.axes), convert(BStarts, b.block_starts), convert(BStrides,b.block_strides), convert(LL,b.l), convert(UU,b.u))

colblockbandwidths(bs::BlockSkylineSizes) = (bs.l, bs.u)

blockstart(block_sizes::BlockSkylineSizes, K, J) = block_sizes.block_starts[K,J]
blockstride(block_sizes::BlockSkylineSizes, J) = block_sizes.block_strides[J]

blockstart(A::AbstractMatrix, K, J) = blockstart(A.block_sizes,K,J)
blockstride(A::AbstractMatrix, J) = blockstride(A.block_sizes,J)

blockbandwidths(bs::BlockSkylineSizes) = maximum.(colblockbandwidths(bs))
blockbandwidth(bs::BlockSkylineSizes, i::Int) = blockbandwidths(bs)[i]

size(bs::BlockSkylineSizes) = map(length,bs.axes)

==(A::BlockSkylineSizes, B::BlockSkylineSizes) = blockisequal(A.axes,B.axes) && A.block_starts == B.block_starts &&
    A.l == B.l && A.u == B.u

colrange(B::BlockSkylineSizes, J::Integer) = max(1, J-B.u[J]):min(blocklength(B.axes[1]), J+B.l[J])

function bb_numentries(B::BlockSkylineSizes)
    axes = B.axes
    numentries = 0
    N = blocklength(axes[1])
    for J = blockaxes(axes[2],1)
        KR = colrange(B, Int(J))
        num_rows = length(axes[1][Block.(KR)])
        num_cols = length(axes[2][J])
        numentries += num_rows*num_cols
    end
    numentries
end

"""
    BlockSkylineMatrix{T,LL,UU}(M::Union{UndefInitializer,UniformScaling,AbstractMatrix},
                                rows, cols, (l::LL, u::UU))

returns a `sum(rows)`×`sum(cols)` block-banded matrix `A` having elements of type `T`,
with block-bandwidths `(l,u)`, and where `A[Block(K,J)]` is a
`Matrix{T}` of size `rows[K]`×`cols[J]`.

`(l,u)` may be integers for constant bandwidths, or integer vectors of length
`length(cols)` for ragged bands. In the latter case, `l` and `u` represent the
number of sub and super-block-bands in each column.

# Examples

```jldoctest
julia> using LinearAlgebra, FillArrays

julia> BlockSkylineMatrix(I, [2,2,2,4], [1,2,3], ([2,0,1],[0,1,1]))
4×3-blocked 10×6 BlockSkylineMatrix{Bool, Vector{Bool}, BlockBandedMatrices.BlockSkylineSizes{Tuple{BlockArrays.BlockedUnitRange{Vector{Int64}}, BlockArrays.BlockedUnitRange{Vector{Int64}}}, Vector{Int64}, Vector{Int64}, BandedMatrices.BandedMatrix{Int64, Matrix{Int64}, Base.OneTo{Int64}}, Vector{Int64}}}:
 1  │  0  0  │  ⋅  ⋅  ⋅
 0  │  1  0  │  ⋅  ⋅  ⋅
 ───┼────────┼─────────
 0  │  0  1  │  0  0  0
 0  │  0  0  │  1  0  0
 ───┼────────┼─────────
 0  │  ⋅  ⋅  │  0  1  0
 0  │  ⋅  ⋅  │  0  0  1
 ───┼────────┼─────────
 ⋅  │  ⋅  ⋅  │  0  0  0
 ⋅  │  ⋅  ⋅  │  0  0  0
 ⋅  │  ⋅  ⋅  │  0  0  0
 ⋅  │  ⋅  ⋅  │  0  0  0

julia> BlockSkylineMatrix(Ones(9,6), [2,3,4], [1,2,3], ([2,0,0],[0,1,1]))
3×3-blocked 9×6 BlockSkylineMatrix{Float64, Vector{Float64}, BlockBandedMatrices.BlockSkylineSizes{Tuple{BlockArrays.BlockedUnitRange{Vector{Int64}}, BlockArrays.BlockedUnitRange{Vector{Int64}}}, Vector{Int64}, Vector{Int64}, BandedMatrices.BandedMatrix{Int64, Matrix{Int64}, Base.OneTo{Int64}}, Vector{Int64}}}:
 1.0  │  1.0  1.0  │   ⋅    ⋅    ⋅
 1.0  │  1.0  1.0  │   ⋅    ⋅    ⋅
 ─────┼────────────┼───────────────
 1.0  │  1.0  1.0  │  1.0  1.0  1.0
 1.0  │  1.0  1.0  │  1.0  1.0  1.0
 1.0  │  1.0  1.0  │  1.0  1.0  1.0
 ─────┼────────────┼───────────────
 1.0  │   ⋅    ⋅   │  1.0  1.0  1.0
 1.0  │   ⋅    ⋅   │  1.0  1.0  1.0
 1.0  │   ⋅    ⋅   │  1.0  1.0  1.0
 1.0  │   ⋅    ⋅   │  1.0  1.0  1.0
```
"""
BlockSkylineMatrix

#  A block matrix where only the bands are nonzero
#   isomorphic to BandedMatrix{Matrix{T}}
struct BlockSkylineMatrix{T, DATA<:AbstractVector{T}, BS<:BlockSkylineSizes} <: AbstractBlockBandedMatrix{T}
    data::DATA
    block_sizes::BS

    global function _BlockSkylineMatrix(data::DATA, block_sizes::BS) where {T,DATA<:AbstractVector{T}, BS<:BlockSkylineSizes}
        new{T,DATA,BS}(data, block_sizes)
    end
end

"""
    BlockBandedMatrix

A `BlockBandedMatrix` is a subtype of `BlockMatrix` of `BlockArrays.jl` whose
layout of non-zero blocks is banded.
"""
const BlockBandedMatrix{T} = BlockSkylineMatrix{T, Vector{T}, BlockBandedSizes}

# Auxiliary outer constructors
@inline _BlockBandedMatrix(data::AbstractVector, bs::BlockBandedSizes) =
    _BlockSkylineMatrix(data, bs)

@inline _BlockBandedMatrix(data::AbstractVector, kr::AbstractVector{Int}, jr::AbstractVector{Int}, (l,u)::NTuple{2, Int}) =
    _BlockBandedMatrix(data, BlockBandedSizes(kr,jr, l,u))

@inline BlockSkylineMatrix{T}(::UndefInitializer, block_sizes::BlockSkylineSizes) where T =
    _BlockSkylineMatrix(Vector{T}(undef, bb_numentries(block_sizes)), block_sizes)

@inline BlockBandedMatrix{T}(::UndefInitializer, block_sizes::BlockBandedSizes) where T =
    _BlockSkylineMatrix(Vector{T}(undef, bb_numentries(block_sizes)), block_sizes)

@inline BlockBandedMatrix{T}(::UndefInitializer, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2, Int}) where T =
    BlockSkylineMatrix{T}(undef, BlockBandedSizes(axes, lu...))

@inline BlockSkylineMatrix{T}(::UndefInitializer, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2, AbstractVector{Int}}) where T =
    BlockSkylineMatrix{T}(undef, BlockSkylineSizes(axes, lu...))

"""
    BlockBandedMatrix{T}(undef, rows::AbstractVector{Int}, cols::AbstractVector{Int},
                        (l,u)::NTuple{2,Int})

Return an unitialized `sum(rows) × sum(cols)` `BlockBandedMatrix` having `eltype` `T`,
with `rows` by `cols` blocks and `(l,u)` as the block-bandwidth.
"""
@inline BlockBandedMatrix{T}(::UndefInitializer, rdims::AbstractVector{Int}, cdims::AbstractVector{Int}, lu::NTuple{2, Int}) where T =
    BlockSkylineMatrix{T}(undef, BlockBandedSizes(rdims, cdims, lu...))

@inline BlockSkylineMatrix{T}(::UndefInitializer, rdims::AbstractVector{Int}, cdims::AbstractVector{Int}, lu::NTuple{2, AbstractVector{Int}}) where T =
    BlockSkylineMatrix{T}(undef, BlockSkylineSizes(rdims, cdims, lu...))

function BlockSkylineMatrix{T}(A::AbstractMatrix, block_sizes::BlockSkylineSizes) where T
    ret = BlockSkylineMatrix(Zeros{T}(size(A)), block_sizes)
    for J = blockaxes(ret,2), K = blockcolsupport(ret, J)
        kr, jr = getindex.(block_sizes.axes, (K, J))
        view(ret, K, J) .= view(A, kr, jr)
    end
    ret
end

function BlockSkylineMatrix{T}(A::AbstractBlockBandedMatrix, block_sizes::BlockSkylineSizes) where T
    ret = BlockSkylineMatrix(Zeros{T}(size(A)), block_sizes)
    blockisequal(axes(A), block_sizes.axes) || throw(ArgumentError())
    for J = blockaxes(ret,2), K = blockcolsupport(ret, J)
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
    if size(Z) ≠ map(length,block_sizes.axes)
        throw(DimensionMismatch("Size of input $(size(Z)) must be consistent with $(size(block_sizes))"))
    end
    _BlockSkylineMatrix(zeros(T, bb_numentries(block_sizes)), block_sizes)
end


function BlockSkylineMatrix{T}(E::Eye, block_sizes::BlockSkylineSizes) where T
    if size(E) ≠ size(block_sizes)
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
                           rdims::AbstractVector{Int}, cdims::AbstractVector{Int}, lu::NTuple{2,AbstractVector{Int}}) where T =
                               BlockSkylineMatrix{T}(A, BlockSkylineSizes(rdims, cdims, lu...))

BlockBandedMatrix{T}(A::Union{AbstractMatrix,UniformScaling},
                           rdims::AbstractVector{Int}, cdims::AbstractVector{Int}, lu::NTuple{2,Int}) where T =
                               BlockSkylineMatrix{T}(A, BlockBandedSizes(rdims, cdims, lu...))

BlockSkylineMatrix(A::Union{AbstractMatrix,UniformScaling},
                        rdims::AbstractVector{Int}, cdims::AbstractVector{Int},
                        lu::NTuple{2,AbstractVector{Int}}) = BlockSkylineMatrix{eltype(A)}(A, rdims, cdims, lu)


"""
    BlockBandedMatrix(A::Union{AbstractMatrix,UniformScaling},
                        rows::AbstractVector{Int}, cols::AbstractVector{Int},
                        (l,u)::NTuple{2,Int})

Return a `sum(rows) × sum(cols)` `BlockBandedMatrix`, with `rows` by `cols` blocks,
with `(l,u)` as the block-bandwidth.
The structural non-zero entries are equal to the corresponding indices of `A`.

# Examples
```jldoctest
julia> using LinearAlgebra, FillArrays

julia> l,u = 0,1; # block bandwidths

julia> nrowblk, ncolblk = 3, 3; # number of row/column blocks

julia> rows = 1:nrowblk; cols = 1:ncolblk; # block sizes

julia> BlockBandedMatrix(I, rows, cols, (l,u))
3×3-blocked 6×6 BlockBandedMatrix{Bool}:
 1  │  0  0  │  ⋅  ⋅  ⋅
 ───┼────────┼─────────
 ⋅  │  1  0  │  0  0  0
 ⋅  │  0  1  │  0  0  0
 ───┼────────┼─────────
 ⋅  │  ⋅  ⋅  │  1  0  0
 ⋅  │  ⋅  ⋅  │  0  1  0
 ⋅  │  ⋅  ⋅  │  0  0  1

julia> BlockBandedMatrix(Ones(sum(rows),sum(cols)), rows, cols, (l,u))
3×3-blocked 6×6 BlockBandedMatrix{Float64}:
 1.0  │  1.0  1.0  │   ⋅    ⋅    ⋅
 ─────┼────────────┼───────────────
  ⋅   │  1.0  1.0  │  1.0  1.0  1.0
  ⋅   │  1.0  1.0  │  1.0  1.0  1.0
 ─────┼────────────┼───────────────
  ⋅   │   ⋅    ⋅   │  1.0  1.0  1.0
  ⋅   │   ⋅    ⋅   │  1.0  1.0  1.0
  ⋅   │   ⋅    ⋅   │  1.0  1.0  1.0
```
"""
BlockBandedMatrix(A::Union{AbstractMatrix,UniformScaling},
                        rdims::AbstractVector{Int}, cdims::AbstractVector{Int},
                        lu::NTuple{2,Int}) = BlockBandedMatrix{eltype(A)}(A, rdims, cdims, lu)

"""
    BlockBandedMatrix(A::AbstractMatrix, (l,u)::NTuple{2,Int})

Return a `BlockBandedMatrix` with block-bandwidths `(l,u)`, where the
structural non-zero blocks correspond to those of `A`.

Examples
```jldoctest
julia> using BlockArrays

julia> B = BlockArray(ones(6,6), 1:3, 1:3);

julia> BlockBandedMatrix(B, (1,1))
3×3-blocked 6×6 BlockBandedMatrix{Float64}:
 1.0  │  1.0  1.0  │   ⋅    ⋅    ⋅
 ─────┼────────────┼───────────────
 1.0  │  1.0  1.0  │  1.0  1.0  1.0
 1.0  │  1.0  1.0  │  1.0  1.0  1.0
 ─────┼────────────┼───────────────
  ⋅   │  1.0  1.0  │  1.0  1.0  1.0
  ⋅   │  1.0  1.0  │  1.0  1.0  1.0
  ⋅   │  1.0  1.0  │  1.0  1.0  1.0
```
"""
BlockBandedMatrix(A::AbstractMatrix, lu::NTuple{2,Int}) = BlockBandedMatrix(A, BlockBandedSizes(axes(A), lu...))

function convert(::Type{BlockSkylineMatrix}, A::AbstractMatrix)
    block_sizes = BlockSkylineSizes(axes(A), colblockbandwidths(A)...)

    copyto!(BlockSkylineMatrix{eltype(A)}(undef, block_sizes), A)
end

function convert(::Type{BlockBandedMatrix}, A::AbstractMatrix)
    convert(BlockSkylineMatrix, A)
end


BlockSkylineMatrix(A::AbstractMatrix) = convert(BlockSkylineMatrix, A)
BlockBandedMatrix(A::AbstractMatrix) = convert(BlockBandedMatrix, A)

similar(A::BlockSkylineMatrix, T::Type=eltype(A), bs::BlockSkylineSizes=A.block_sizes) =
    BlockSkylineMatrix{T}(undef, bs)

axes(A::BlockSkylineMatrix) = A.block_sizes.axes

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

zeroblock(A::BlockSkylineMatrix, K::Int, J::Int) =
    Matrix(Zeros{eltype(A)}(length.(getindex.(axes(A),(Block(K),Block(J))))))


###########################
# AbstractArray Interface #
###########################

# @inline function Base.similar(block_array::BlockSkylineMatrix{T}, ::Type{T2}) where {T,N,T2}
#     BlockArray(similar(block_array.blocks, Array{T2, N}), copy(block_array.block_sizes))
# end

Base.size(arr::BlockSkylineMatrix) =
    @inbounds return map(length,axes(arr))


@inline function getindex(A::BlockSkylineMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    bi = findblockindex.(axes(A), (i,j))
    @inbounds v = view(A, block.(bi)...)[blockindex.(bi)...]
    return v
end

@inline function setindex!(A::BlockSkylineMatrix{T}, v, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    bi = findblockindex.(axes(A), (i,j))
    V = view(A, block.(bi)...)
    @inbounds V[blockindex.(bi)...] = convert(T, v)::T
    return A
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




##################
# BlockBandedBlock
#
#   views of the blocks satisfy the Matrix interface, and are memory-compatible
#   with StridedMatrix.
##################

const BlockBandedBlock{T} = SubArray{T,2,<:BlockSkylineMatrix,<:Tuple{BlockSlice1,BlockSlice1},false}




# gives the columns of parent(V).data that encode the block
_parent_blocks(V::BlockBandedBlock)::Tuple{Int,Int} =
    first(first(parentindices(V)).block.n),first(last(parentindices(V)).block.n)

######################################
# Matrix interface  for Blocks #
######################################


MemoryLayout(::Type{<:BlockBandedBlock}) = ColumnMajor()
Base.elsize(::Type{<:BlockSkylineMatrix{T,R}}) where {T,R} = Base.elsize(R)

function Base.unsafe_convert(::Type{Ptr{T}}, V::BlockBandedBlock{T}) where T
    A = parent(V)
    K,J = _parent_blocks(V)
    Base.unsafe_convert(Ptr{T}, A.data) + sizeof(T)*(blockstart(A,K,J)-1)
end

strides(V::BlockBandedBlock) = (1,parent(V).block_sizes.block_strides[_parent_blocks(V)[2]])

@propagate_inbounds function getindex(V::BlockBandedBlock, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = _parent_blocks(V)
    if -A.block_sizes.l[J] ≤ J-K ≤ A.block_sizes.u[J]
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
    K,J = _parent_blocks(V)
    if -A.block_sizes.l[J] ≤ J-K ≤ A.block_sizes.u[J]
        b_start = blockstart(A,K,J)
        # TODO: What to do if b_start == 0 ?
        b_stride = A.block_sizes.block_strides[J]
        A.data[b_start + k-1 + (j-1)*b_stride ] = v
    elseif !iszero(v) # allow setindex for 0 data
        throw(BandError(A, J-K))
    end
    return V
end

"""
    copy_accommodating_diagonals(A::BlockSkylineMatrix, diagonals::UnitRange{<:Integer})

Return copy of `A`, ensuring blocks are present covering the
`diagonals` as well.
"""
function copy_accommodating_diagonals(A::BlockSkylineMatrix, diagonals::UnitRange{<:Integer}, ::Type{T}=eltype(A)) where T
    checksquareblocks(A)
    bs = A.block_sizes
    l,u = bs.l,bs.u
    ax = first(bs.axes)

    if 0 ∈ diagonals
        l = max.(l, 0)
        u = max.(u, 0)
    end

    for d = extrema(diagonals)
        d == 0 && continue # Already taken care of above
        v = d > 0 ? u : l
        # The j:th element of rows is the row index which the diagonal
        # d covers in the j:th column.
        rows = clamp.(ax .- d, 1, last(ax))
        for (j,i) in enumerate(rows)
            # First we find which block the j:th element of the main
            # diagonal would occupy.
            md_block = searchsortedfirst(blocklasts(ax), j)

            # Next we find which block covers row i
            d_block = searchsortedfirst(blocklasts(ax), i)

            # Finally, we increase the block-bandwidth as necessary
            v[md_block] = max(v[md_block], abs(d_block-md_block))
        end
    end

    BlockSkylineMatrix{T}(A, BlockSkylineSizes((ax,ax), l, u))
end

for op in (:-, :+)
    @eval begin
        function $op(A::BlockSkylineMatrix, I::UniformScaling)
            B = copy_accommodating_diagonals(A, 0:0, Base._return_type(+, Tuple{eltype(A), eltype(I)}))
            @inbounds for i in axes(A, 1)
                B[i,i] = $op(B[i,i], I.λ)
            end
            B
        end
        function $op(I::UniformScaling, A::BlockSkylineMatrix)
            B = copy_accommodating_diagonals($op(A), 0:0, Base._return_type(+, Tuple{eltype(A), eltype(I)}))
            @inbounds for i in axes(A, 1)
                B[i,i] += I.λ
            end
            B
        end

        function $op(A::BlockSkylineMatrix, D::Diagonal)
            B = copy_accommodating_diagonals(A, 0:0, Base._return_type(+, Tuple{eltype(A), eltype(D)}))
            @inbounds for i in axes(A, 1)
                B[i,i] = $op(B[i,i], D.diag[i])
            end
            B
        end
        function $op(D::Diagonal, A::BlockSkylineMatrix)
            B = copy_accommodating_diagonals($op(A), 0:0, Base._return_type(+, Tuple{eltype(A), eltype(D)}))
            @inbounds for i in axes(A, 1)
                B[i,i] += D.diag[i]
            end
            B
        end

        function $op(A::BlockSkylineMatrix, Bd::Bidiagonal)
            B = copy_accommodating_diagonals(A, Bd.uplo == 'U' ? (0:1) : (-1:0),
                                             Base._return_type(+, Tuple{eltype(A), eltype(Bd)}))
            @inbounds for i in axes(A, 1)
                B[i,i] = $op(B[i,i], Bd.dv[i])
            end
            @inbounds for i in 1:size(A, 1)-1
                Bd.uplo == 'U' && (B[i,i+1] = $op(B[i,i+1], Bd.ev[i]))
                Bd.uplo == 'L' && (B[i+1,i] = $op(B[i+1,i], Bd.ev[i]))
            end
            B
        end
        function $op(Bd::Bidiagonal, A::BlockSkylineMatrix)
            B = copy_accommodating_diagonals($op(A), Bd.uplo == 'U' ? (0:1) : (-1:0),
                                             Base._return_type(+, Tuple{eltype(A), eltype(Bd)}))
            @inbounds for i in axes(A, 1)
                B[i,i] += Bd.dv[i]
            end
            @inbounds for i in 1:size(A, 1)-1
                Bd.uplo == 'U' && (B[i,i+1] += Bd.ev[i])
                Bd.uplo == 'L' && (B[i+1,i] += Bd.ev[i])
            end
            B
        end

        function $op(A::BlockSkylineMatrix, T::Tridiagonal)
            B = copy_accommodating_diagonals(A, -1:1, Base._return_type(+, Tuple{eltype(A), eltype(T)}))
            @inbounds for i in axes(A, 1)
                B[i,i] = $op(B[i,i], T.d[i])
            end
            @inbounds for i in 1:size(A, 1)-1
                B[i,i+1] = $op(B[i,i+1], T.du[i])
                B[i+1,i] = $op(B[i+1,i], T.dl[i])
            end
            B
        end
        function $op(T::Tridiagonal, A::BlockSkylineMatrix)
            B = copy_accommodating_diagonals($op(A), -1:1, Base._return_type(+, Tuple{eltype(A), eltype(T)}))
            @inbounds for i in axes(A, 1)
                B[i,i] += T.d[i]
            end
            @inbounds for i in 1:size(A, 1)-1
                B[i,i+1] += T.du[i]
                B[i+1,i] += T.dl[i]
            end
            B
        end

        function $op(A::BlockSkylineMatrix, T::SymTridiagonal)
            B = copy_accommodating_diagonals(A, -1:1, Base._return_type(+, Tuple{eltype(A), eltype(T)}))
            @inbounds for i in axes(A, 1)
                B[i,i] = $op(B[i,i], T.dv[i])
            end
            @inbounds for i in 1:size(A, 1)-1
                B[i,i+1] = $op(B[i,i+1], T.ev[i])
                B[i+1,i] = $op(B[i+1,i], T.ev[i])
            end
            B
        end
        function $op(T::SymTridiagonal, A::BlockSkylineMatrix)
            B = copy_accommodating_diagonals($op(A), -1:1, Base._return_type(+, Tuple{eltype(A), eltype(T)}))
            @inbounds for i in axes(A, 1)
                B[i,i] += T.dv[i]
            end
            @inbounds for i in 1:size(A, 1)-1
                B[i,i+1] += T.ev[i]
                B[i+1,i] += T.ev[i]
            end
            B
        end
    end
end
