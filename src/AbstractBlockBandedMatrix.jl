
"""
    AbstractBlockBandedLayout

isa a `MemoryLayout` that indicates that the array implements the block-banded
interface.
"""
abstract type AbstractBlockBandedLayout <: AbstractBlockLayout end

"""
    AbstractBandedBlockBandedLayout

isa a `MemoryLayout` that indicates that the array implements the banded-block-banded
interface.
"""
abstract type AbstractBandedBlockBandedLayout <: AbstractBlockBandedLayout end


struct BandedBlockBandedLayout <: AbstractBandedBlockBandedLayout end
struct BlockBandedLayout <: AbstractBlockBandedLayout end

struct BandedBlockBandedColumns{LAY} <: AbstractBandedBlockBandedLayout end
struct BandedBlockBandedRows{LAY} <: AbstractBandedBlockBandedLayout end
struct BlockBandedColumns{LAY} <: AbstractBlockBandedLayout end
struct BlockBandedRows{LAY} <: AbstractBlockBandedLayout end

const BandedBlockBandedColumnMajor = BandedBlockBandedColumns{ColumnMajor}
const BandedBlockBandedRowMajor = BandedBlockBandedRows{ColumnMajor}
const BlockBandedColumnMajor = BlockBandedColumns{ColumnMajor}
const BlockBandedRowMajor = BlockBandedRows{ColumnMajor}

transposelayout(::BandedBlockBandedColumns{Lay}) where Lay = BandedBlockBandedRows{Lay}()
transposelayout(::BandedBlockBandedRows{Lay}) where Lay = BandedBlockBandedColumns{Lay}()
transposelayout(::BlockBandedColumns{Lay}) where Lay = BlockBandedRows{Lay}()
transposelayout(::BlockBandedRows{Lay}) where Lay = BlockBandedColumns{Lay}()

conjlayout(::Type{T}, ::BandedBlockBandedColumns{Lay}) where {T<:Complex,Lay} = BandedBlockBandedColumns{typeof(conjlayout(T,Lay))}()
conjlayout(::Type{T}, ::BandedBlockBandedRows{Lay}) where {T<:Complex,Lay} = BandedBlockBandedRows{typeof(conjlayout(T,Lay))}()
conjlayout(::Type{T}, ::BlockBandedColumns{Lay}) where {T<:Complex,Lay} = BlockBandedColumns{typeof(conjlayout(T,Lay))}()
conjlayout(::Type{T}, ::BlockBandedRows{Lay}) where {T<:Complex,Lay} = BlockBandedRows{typeof(conjlayout(T,Lay))}()

symmetriclayout(::AbstractBandedBlockBandedLayout) = BandedBlockBandedLayout()
hermitianlayout(_, ::AbstractBandedBlockBandedLayout) = BandedBlockBandedLayout()

function blockbandwidths(S::HermOrSym)
    l, u = blockbandwidths(parent(S))
    if S.uplo == 'U'
        (u, u)
    else
        (l, l)
    end
end
function subblockbandwidths(S::HermOrSym)
    m = max(subblockbandwidths(parent(S))...)
    (m,m)
end

# AbstractBandedMatrix must implement

# A BlockBandedMatrix is a BlockMatrix, but is not a BandedMatrix
abstract type AbstractBlockBandedMatrix{T} <: AbstractBlockMatrix{T} end
MemoryLayout(::Type{<:AbstractBlockBandedMatrix}) = BlockBandedLayout()


"""
    blockbandwidths(A)

Returns a tuple containing the upper and lower blockbandwidth of `A`.
"""
blockbandwidths(A::AbstractVecOrMat) = blockbandwidths(MemoryLayout(A), axes(A), A)
blockbandwidths(_, _, A) = (blocksize(A,1)-1 , blocksize(A,2)-1)

"""
    blockbandwidth(A,i)

Returns the lower blockbandwidth (`i==1`) or the upper blockbandwidth (`i==2`).
"""
blockbandwidth(A, k::Integer) = blockbandwidths(A)[k]

"""
    blockbandrange(A)

Returns the range `-blockbandwidth(A,1):blockbandwidth(A,2)`.
"""
blockbandrange(A::AbstractMatrix) = -blockbandwidth(A,1):blockbandwidth(A,2)



# start/stop indices of the i-th column/row, bounded by actual matrix size
@inline blockbanded_blockcolstart(A, i::Block{1}) = Block(max(Int(i)-colblockbandwidth(A,2)[Int(i)], 1))
@inline blockbanded_blockcolstop(A, i::Block{1}) = Block(max(min(Int(i)+colblockbandwidth(A,1)[Int(i)], blocksize(A, 1)), 0))
@inline blockbanded_blockrowstart(A, i::Block{1}) = Block(max(Int(i)-rowblockbandwidth(A,1)[Int(i)], 1))
@inline blockbanded_blockrowstop(A, i::Block{1}) = Block(max(min(Int(i)+rowblockbandwidth(A,2)[Int(i)], blocksize(A, 2)), 0))

blockbanded_blockcolstart(A, i::BlockRange) = blockbanded_blockcolstart(A, minimum(i))
blockbanded_blockrowstart(A, i::BlockRange) = blockbanded_blockrowstart(A, minimum(i))
blockbanded_blockcolstop(A, i::BlockRange) = blockbanded_blockcolstop(A, maximum(i))
blockbanded_blockrowstop(A, i::BlockRange) = blockbanded_blockrowstop(A, maximum(i))

const AllBlockBandedLayout{UPLO,UNIT} = Union{AbstractBlockBandedLayout,TriangularLayout{UPLO,UNIT,<:AbstractBlockBandedLayout},
                                                SymmetricLayout{<:AbstractBlockBandedLayout}, HermitianLayout{<:AbstractBlockBandedLayout}}

@inline blockcolsupport(::AllBlockBandedLayout, A, i) = isempty(i) ? (Block(1):Block(0)) : blockbanded_blockcolstart(A,i):blockbanded_blockcolstop(A,i)
@inline blockrowsupport(::AllBlockBandedLayout, A, i) = isempty(i) ? (Block(1):Block(0)) : blockbanded_blockrowstart(A,i):blockbanded_blockrowstop(A,i)


# length of i-the column/row
@inline blockcollength(A, i) = max(Int(blockcolstop(A, i)) - Int(blockcolstart(A, i)) + 1, 0)
@inline blockrowlength(A, i) = max(Int(blockrowstop(A, i)) - Int(blockrowstart(A, i)) + 1, 0)

# this gives the block bandwidth in each block column/row
@inline colblockbandwidths(A::AbstractMatrix) = Fill.(blockbandwidths(A), blocksize(A,2))
@inline rowblockbandwidths(A::AbstractMatrix) = Fill.(blockbandwidths(A), blocksize(A,1))

@inline colblockbandwidth(bs, i::Int) = colblockbandwidths(bs)[i]
@inline rowblockbandwidth(bs, i::Int) = rowblockbandwidths(bs)[i]

"""
    isblockbanded(A)

returns true if a matrix implements the block banded interface.
"""
isblockbanded(::AbstractBlockBandedMatrix) = true
isblockbanded(_) = false

# override bandwidth(A,k) for each AbstractBandedMatrix
# override inbands_getindex(A,k,j)


# return id of first empty diagonal intersected along row k
function _firstblockdiagrow(A::AbstractMatrix, k::Int)
    a, b = blockrowstart(A, k), blockrowstop(A, k)
    c = a == 1 ? b+1 : a-1
    c-k
end

# return id of first empty diagonal intersected along column j
function _firstblockdiagcol(A::AbstractMatrix, j::Int)
    a, b = blockcolstart(A, j), blockcolstop(A, j)
    r = a == 1 ? b+1 : a-1
    j-r
end




######################################
#  RaggedMatrix interface
######################################

@inline function blockbanded_colstart(A, i::Integer)
    bs = blockcolstart(A,findblock(axes(A,2),i))
    if isempty(axes(A,1))
        1
    elseif Int(bs) ≤ blocksize(A, 1)
        first(axes(A,1)[bs])
    else
        size(A,1)+1
    end
end

@inline function blockbanded_colstop(A, i::Integer)
    CS = blockcolstop(A,findblock(axes(A,2),i))
    CS in blockaxes(axes(A,1), 1) || return 0
    last(axes(A,1)[CS])
end

@inline function blockbanded_rowstart(A, i::Integer)
    bs = blockrowstart(A,findblock(axes(A,1),i))
    if isempty(axes(A,2))
        1
    elseif Int(bs) ≤ blocksize(A, 2)
        first(axes(A,2)[bs])
    else
        size(A,2)+1
    end
end

@inline function blockbanded_rowstop(A, i::Integer)
    CS = blockrowstop(A,findblock(axes(A,1),i))
    CS in blockaxes(axes(A,2), 1) || return 0
    last(axes(A,2)[CS])
end

@inline blockbanded_colsupport(A, j::Integer) = blockbanded_colstart(A, j):blockbanded_colstop(A, j)
@inline blockbanded_rowsupport(A, j::Integer) = blockbanded_rowstart(A, j):blockbanded_rowstop(A, j)

@inline blockbanded_rowsupport(A, j) = isempty(j) ? (1:0) : blockbanded_rowstart(A,minimum(j)):blockbanded_rowstop(A,maximum(j))
@inline blockbanded_colsupport(A, j) = isempty(j) ? (1:0) : blockbanded_colstart(A,minimum(j)):blockbanded_colstop(A,maximum(j))

@inline colsupport(::AbstractBlockBandedLayout, A, j) = blockbanded_colsupport(A, j)
@inline rowsupport(::AbstractBlockBandedLayout, A, j) = blockbanded_rowsupport(A, j)



# default implementation loops over all indices, including zeros
function fill!(A::AbstractBlockBandedMatrix, val::Any)
  iszero(val) || throw(BandError(A))
  fill!(A.data, val)
  A
end


function ==(A::AbstractBlockBandedMatrix, B::AbstractBlockBandedMatrix)
    axes(A) == axes(B) || return false
    blockisequal(axes(A), axes(B)) || return Base.invoke(==, Tuple{AbstractArray,AbstractArray}, A, B)
    l,u = max.(blockbandwidths(A), blockbandwidths(B))
    N = blocksize(A,1)
    for J = blockaxes(A,2), K = max(Block(1),J-u):min(J+l,Block(N))
        view(A, K, J) == view(B, K, J) || return false
    end
    return true
end
