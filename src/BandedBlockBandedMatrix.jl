

function check_data_sizes(data::AbstractMatrix, raxis, (l,u), (λ,μ))
    if blocksize(data,1) ≠ l + u + 1 && !(blocksize(data,1) == 0 && (-l > u || -λ > μ))
        throw(ArgumentError("Data matrix must have number of row blocks equal to number of block bands"))
    end
    for K = blockaxes(data,1)
        if length(axes(data,1)[K]) ≠ λ + μ + 1 && !(-λ > μ)
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

const DefaultBandedBlockBandedMatrix{T} = BandedBlockBandedMatrix{T, PseudoBlockMatrix{T, Matrix{T}, NTuple{2,DefaultBlockAxis}}, DefaultBlockAxis}

@inline _BandedBlockBandedMatrix(data::AbstractMatrix, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) = 
    _BandedBlockBandedMatrix(PseudoBlockArray(data,(blockedrange(Fill(sum(λμ)+1,sum(lu)+1)),axes[2])), axes[1], lu, λμ)    

_BandedBlockBandedMatrix(data::AbstractMatrix,rblocksizes::AbstractVector{Int}, cblocksizes::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) =
    _BandedBlockBandedMatrix(data, (blockedrange(rblocksizes),blockedrange(cblocksizes)), lu, λμ)

_blocklengths2blocklasts(b::Fill) = iszero(FillArrays.getindex_value(b)) ? (1:1:0) : cumsum(b)
_bbb_data_axes(caxes, lu, λμ) = (blockedrange(Fill(max(0,sum(λμ)+1),max(0,sum(lu)+1))),caxes)

BandedBlockBandedMatrix{T,B,R}(::UndefInitializer, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B,R<:AbstractUnitRange{Int}} =
    _BandedBlockBandedMatrix(B(undef, _bbb_data_axes(axes[2],lu,λμ)), axes[1], lu, λμ)

BandedBlockBandedMatrix{T,B,R}(::UndefInitializer, rblocksizes::AbstractVector{Int}, cblocksizes::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B,R<:AbstractUnitRange{Int}} =
    BandedBlockBandedMatrix{T,B,R}(undef, (blockedrange(rblocksizes),blockedrange(cblocksizes)), lu, λμ)

BandedBlockBandedMatrix{T,B}(::UndefInitializer, axes::NTuple{2,AbstractUnitRange{Int}}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B} =
    BandedBlockBandedMatrix{T,B,typeof(axes[1])}(undef,axes,lu,λμ)

BandedBlockBandedMatrix{T,B}(::UndefInitializer, rblocksizes::AbstractVector{Int}, cblocksizes::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) where {T,B} =
    BandedBlockBandedMatrix{T,B}(undef, (blockedrange(rblocksizes),blockedrange(cblocksizes)), lu, λμ)


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


BandedBlockBandedMatrix{T}(A::AbstractMatrix) where T =
    copyto!(BandedBlockBandedMatrix{T}(undef, axes(A), blockbandwidths(A), subblockbandwidths(A)), A)

BandedBlockBandedMatrix(A::AbstractMatrix{T}) where T = BandedBlockBandedMatrix{T}(A)

BandedBlockBandedMatrix(A::AbstractMatrix, rdims::AbstractVector{Int}, cdims::AbstractVector{Int}, lu::NTuple{2,Int}, λμ::NTuple{2,Int}) =
    BandedBlockBandedMatrix(A, (blockedrange(rdims), blockedrange(cdims)), lu, λμ)

similar(A::BandedBlockBandedMatrix, ::Type{T}, axes::NTuple{2,AbstractUnitRange{Int}}) where T =
    BandedBlockBandedMatrix{T}(undef, axes, blockbandwidths(A), subblockbandwidths(A))

@inline similar(A::BandedBlockBandedMatrix, ::Type{T}, axes::Tuple{BlockedUnitRange,AbstractUnitRange{Int}}) where T =
    BandedBlockBandedMatrix{T}(undef, axes, blockbandwidths(A), subblockbandwidths(A))
@inline similar(A::BandedBlockBandedMatrix, ::Type{T}, axes::Tuple{AbstractUnitRange{Int},BlockedUnitRange}) where T =
    BandedBlockBandedMatrix{T}(undef, axes, blockbandwidths(A), subblockbandwidths(A))    
@inline similar(A::BandedBlockBandedMatrix, ::Type{T}, axes::Tuple{BlockedUnitRange,BlockedUnitRange}) where T =
    BandedBlockBandedMatrix{T}(undef, axes, blockbandwidths(A), subblockbandwidths(A))



similar(A::BandedBlockBandedMatrix{T}, axes::NTuple{2,AbstractUnitRange{Int}}) where T =
    similar(Matrix{T}, map(length,axes)...)

axes(A::BandedBlockBandedMatrix) = (A.raxis, axes(A.data,2))

function sparse(A::BandedBlockBandedMatrix)
    i = Vector{Int}()
    j = Vector{Int}()
    z = Vector{eltype(A)}()
    for J = blockaxes(A,2), K = blockcolsupport(A, J)
        B = view(A, K, J)
        ĩ = _banded_rowval(B)
        j̃ = _banded_colval(B)
        z̃ = _banded_nzval(B)
        ĩ .+= first(axes(A,1)[K])-1
        j̃ .+= first(axes(A,2)[J])-1
        append!(i, ĩ)
        append!(j, j̃)
        append!(z, z̃)
    end
    sparse(i, j, z, size(A)...)
end

################################
# BandedBlockBandedMatrix Interface #
################################

bandedblockbandedcolumns(L::AbstractColumnMajor) = BandedBlockBandedColumnMajor()
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


zeroblock(A::BandedBlockBandedMatrix, K::Int, J::Int) =
    BandedMatrix(Zeros{eltype(A)}(length.(getindex.(axes(A),(Block(K),Block(J))))), (A.λ, A.μ))

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
    if -A.l ≤ Int(block(BJ)-block(BI)) ≤ A.u
        V = view(A, block(BI),block(BJ))
        @inbounds V[blockindex(BI),blockindex(BJ)] = convert(T, v)::T
    elseif !iszero(v)
        throw(BandError(A))
    end
    return v
end

## structured matrix methods ##
function Base.replace_in_print_matrix(A::BandedBlockBandedMatrix, i::Integer, j::Integer, s::AbstractString)
    bi = findblockindex.(axes(A), (i,j))
    I,J = block.(bi)
    i,j = blockindex.(bi)
    -A.l ≤ Int(J-I) ≤ A.u && -A.λ ≤ j-i ≤ A.μ ? s : Base.replace_with_centered_mark(s)
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

const SubBandedBlockBandedMatrix{T,R1,R2} =
    SubArray{T,2,<:BandedBlockBandedMatrix{T},<:Tuple{<:BlockSlice{R1},<:BlockSlice{R2}}}


sublayout(::BandedBlockBandedColumnMajor, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{Block1}}}) = BandedColumnMajor()
sublayout(::BandedBlockBandedColumnMajor, ::Type{<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{BlockIndexRange1}}}) = BandedColumnMajor()
sublayout(::BandedBlockBandedColumnMajor, ::Type{<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{Block1}}}) = BandedColumnMajor()
sublayout(::BandedBlockBandedColumnMajor, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{BlockIndexRange1}}}) = BandedColumnMajor()
sublayout(::BandedBlockBandedColumnMajor, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{BlockRange1}}}) = BandedBlockBandedColumnMajor()
sublayout(::BandedBlockBandedColumnMajor, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{BlockRange1}}}) = BandedBlockBandedColumnMajor()
sublayout(::BandedBlockBandedColumnMajor, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{Block1}}}) = BandedBlockBandedColumnMajor()
sublayout(::BandedBlockBandedColumnMajor, ::Type{<:Tuple{BlockSlice{BlockRange1},BlockSlice{BlockIndexRange1}}}) = BandedBlockBandedColumnMajor()

isbanded(A::SubArray{<:Any,2,<:BandedBlockBandedMatrix}) = MemoryLayout(typeof(A)) == BandedColumnMajor()
isbandedblockbanded(A::SubArray{<:Any,2,<:BandedBlockBandedMatrix}) = MemoryLayout(typeof(A)) == BandedBlockBandedColumnMajor()


subblockbandwidths(V::SubBandedBlockBandedMatrix) = subblockbandwidths(parent(V))

function blockbandwidths(V::SubBandedBlockBandedMatrix{<:Any,BlockRange1,Block1})
    A = parent(V)

    KR = parentindices(V)[1].block.indices[1]
    J = parentindices(V)[2].block
    shift = Int(KR[1])-Int(J)
    blockbandwidth(A,1) - shift, blockbandwidth(A,2) + shift
end

function blockbandwidths(V::SubBandedBlockBandedMatrix{<:Any,Block1,BlockRange1})
    A = parent(V)

    K = parentindices(V)[1].block
    JR = parentindices(V)[2].block.indices[1]
    shift = Int(K)-Int(JR[1])

    blockbandwidth(A,1) - shift, blockbandwidth(A,2) + shift
end

function blockbandwidths(V::SubBandedBlockBandedMatrix{<:Any,BlockRange1,BlockRange1})
    A = parent(V)

    KR = parentindices(V)[1].block.indices[1]
    JR = parentindices(V)[2].block.indices[1]
    shift = Int(first(KR))-Int(first(JR))

    blockbandwidth(A,1) - shift, blockbandwidth(A,2) + shift
end


const BandedBlockBandedBlock{T, BLOCKS, RAXIS} = SubArray{T,2,BandedBlockBandedMatrix{T, BLOCKS, RAXIS},<:Tuple{<:BlockSlice1,<:BlockSlice1},false}


BroadcastStyle(::Type{<: BandedBlockBandedBlock}) = BandedStyle()


function inblockbands(V::SubArray{<:Any,2,<:AbstractMatrix,<:Tuple{<:BlockSlice1,<:BlockSlice1},false})
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    K, J = K_sl.block, J_sl.block
    l,u = blockbandwidths(A)
    -l ≤ Int(J-K) ≤ u
end

function parentblock(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{BlockIndexRange1}}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(A, K_sl.block.block, J_sl.block.block)
end

function parentblock(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{Block1},BlockSlice{BlockIndexRange1}}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(A, K_sl.block, J_sl.block.block)
end

function parentblock(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{Block1}}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(A, K_sl.block.block, J_sl.block)
end

# gives the columns of parent(V).data that encode the block
parentblocks2Int(V::BandedBlockBandedBlock)::Tuple{Int,Int} = Int(first(parentindices(V)).block),
                                                          Int(last(parentindices(V)).block)


######################################
# BandedMatrix interface  for Blocks #
######################################
@inline function bandwidths(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice1,BlockSlice1}}) where T
    inblockbands(V) && return subblockbandwidths(parent(V))
    (-720,-720)
end

function bandwidths(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{BlockIndexRange1}}}) where T
   B = parentblock(V)
   K_sl, J_sl = parentindices(V)
   bandwidths(B) .+ (-1,1) .* bandshift(K_sl.block.indices[1],J_sl.block.indices[1])
end

function bandwidths(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{Block1},BlockSlice{BlockIndexRange1}}}) where T
    B = parentblock(V)
    K_sl, J_sl = parentindices(V)
    bandwidths(B) .+ (-1,1) .* bandshift(Base.OneTo(1),J_sl.block.indices[1])
 end

function bandwidths(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{Block1}}}) where T
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

function bandeddata(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{BlockIndexRange1}}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(bandeddata(parentblock(V)), :, J_sl.block.indices[1])
end

function bandeddata(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{Block1},BlockSlice{BlockIndexRange1}}}) where T
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    view(bandeddata(parentblock(V)), :, J_sl.block.indices[1])
end

function bandeddata(V::SubArray{T,2,<:AbstractMatrix,<:Tuple{BlockSlice{BlockIndexRange1},BlockSlice{Block1}}}) where T
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
    if -A.l ≤ J-K ≤ A.u && -A.λ ≤ j-k ≤ A.μ
        inbands_getindex(V, k, j)
    else
        zero(eltype(V))
    end
end

@propagate_inbounds function setindex!(V::BandedBlockBandedBlock, v, k::Int, j::Int)
    @boundscheck checkbounds(V, k, j)
    A = parent(V)
    K,J = parentblocks2Int(V)
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
*(x::Number, A::BandedBlockBandedMatrix) = _BandedBlockBandedMatrix(x*A.data, axes(A,1), blockbandwidths(A), subblockbandwidths(A))
*(A::BandedBlockBandedMatrix, x::Number) = _BandedBlockBandedMatrix(A.data*x, axes(A,1), blockbandwidths(A), subblockbandwidths(A))
