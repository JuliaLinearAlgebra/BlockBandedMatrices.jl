
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
const BandedBlockBandedRowMajor = BandedBlockBandedColumns{RowMajor}
const BlockBandedColumnMajor = BlockBandedColumns{ColumnMajor}
const BlockBandedRowMajor = BlockBandedColumns{RowMajor}

transposelayout(::BandedBlockBandedColumnMajor) = BandedBlockBandedRowMajor()
transposelayout(::BandedBlockBandedRowMajor) = BandedBlockBandedColumnMajor()
transposelayout(::BlockBandedColumnMajor) = BlockBandedRowMajor()
transposelayout(::BlockBandedRowMajor) = BlockBandedColumnMajor()



# AbstractBandedMatrix must implement

# A BlockBandedMatrix is a BlockMatrix, but is not a BandedMatrix
abstract type AbstractBlockBandedMatrix{T} <: AbstractBlockMatrix{T} end


"""
    blockbandwidths(A)

Returns a tuple containing the upper and lower blockbandwidth of `A`.
"""
blockbandwidths(A::AbstractMatrix) = (blocksize(A,1)-1 , blocksize(A,2)-1)

"""
    blockbandwidth(A,i)

Returns the lower blockbandwidth (`i==1`) or the upper blockbandwidth (`i==2`).
"""
blockbandwidth(A::AbstractMatrix, k::Integer) = blockbandwidths(A)[k]

"""
    bandrange(A)

Returns the range `-blockbandwidth(A,1):blockbandwidth(A,2)`.
"""
blockbandrange(A::AbstractMatrix) = -blockbandwidth(A,1):blockbandwidth(A,2)



# start/stop indices of the i-th column/row, bounded by actual matrix size
@inline blockbanded_blockcolstart(A, i::Integer) = Block(max(i-colblockbandwidth(A,2)[i], 1))
@inline blockbanded_blockcolstop(A, i::Integer) = Block(max(min(i+colblockbandwidth(A,1)[i], blocksize(A, 1)), 0))
@inline blockbanded_blockrowstart(A, i::Integer) = Block(max(i-rowblockbandwidth(A,1)[i], 1))
@inline blockbanded_blockrowstop(A, i::Integer) = Block(max(min(i+rowblockbandwidth(A,2)[i], blocksize(A, 2)), 0))
for Func in (:blockbanded_blockcolstart, :blockbanded_blockcolstop, :blockbanded_blockrowstart, :blockbanded_blockrowstop)
    @eval $Func(A, i::Block{1}) = $Func(A, Int(i))
end

blockbanded_blockcolstart(A, i::BlockRange) = blockbanded_blockcolstart(A, minimum(i))
blockbanded_blockrowstart(A, i::BlockRange) = blockbanded_blockrowstart(A, minimum(i))
blockbanded_blockcolstop(A, i::BlockRange) = blockbanded_blockcolstop(A, maximum(i))
blockbanded_blockrowstop(A, i::BlockRange) = blockbanded_blockrowstop(A, maximum(i))


@inline blockcolsupport(::AbstractBlockBandedLayout, A, i) = blockbanded_blockcolstart(A,i):blockbanded_blockcolstop(A,i)
@inline blockrowsupport(::AbstractBlockBandedLayout, A, i) = blockbanded_blockrowstart(A,i):blockbanded_blockrowstop(A,i)


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
    elseif Int(bs) ≤ blocksize(A, 2)
        first(axes(A,1)[bs])
    else
        size(A,1)+1
    end
end

@inline function blockbanded_colstop(A, i::Integer)
    CS = blockcolstop(A,findblock(axes(A,2),i))
    CS == Block(0) && return 0
    last(axes(A,1)[CS])
end

@inline blockbanded_rowstart(A, i::Integer) =
    first(axes(A,2)[blockrowstart(A,findblock(axes(A,1),i))])

@inline function blockbanded_rowstop(A, i::Integer)
    CS = blockrowstop(A,findblock(axes(A,1),i))
    CS == Block(0) && return 0
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
