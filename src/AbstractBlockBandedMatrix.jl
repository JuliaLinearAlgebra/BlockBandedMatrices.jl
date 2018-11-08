# AbstractBandedMatrix must implement

# A BlockBandedMatrix is a BlockMatrix, but is not a BandedMatrix
abstract type AbstractBlockBandedMatrix{T} <: AbstractBlockMatrix{T} end


"""
    blockbandwidths(A)

Returns a tuple containing the upper and lower blockbandwidth of `A`.
"""
blockbandwidths(A::AbstractMatrix) = (nblocks(A,1)-1 , nblocks(A,2)-1)

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
@inline blockcolstart(A::AbstractVecOrMat, i::Integer) = Block(max(i-colblockbandwidth(A,2)[i], 1))
@inline  blockcolstop(A::AbstractVecOrMat, i::Integer) = Block(max(min(i+colblockbandwidth(A,1)[i], nblocks(A, 1)), 0))
@inline blockrowstart(A::AbstractVecOrMat, i::Integer) = Block(max(i-rowblockbandwidth(A,1)[i], 1))
@inline  blockrowstop(A::AbstractVecOrMat, i::Integer) = Block(max(min(i+rowblockbandwidth(A,2)[i], nblocks(A, 2)), 0))

for Func in (:blockcolstart, :blockcolstop, :blockrowstart, :blockrowstop)
    @eval $Func(A, i::Block{1}) = $Func(A, Int(i))
end

@inline blockcolrange(A::AbstractVecOrMat, i) = blockcolstart(A,i):blockcolstop(A,i)
@inline blockrowrange(A::AbstractVecOrMat, i) = blockrowstart(A,i):blockrowstop(A,i)


# length of i-the column/row
@inline blockcollength(A::AbstractVecOrMat, i) = max(Int(blockcolstop(A, i)) - Int(blockcolstart(A, i)) + 1, 0)
@inline blockrowlength(A::AbstractVecOrMat, i) = max(Int(blockrowstop(A, i)) - Int(blockrowstart(A, i)) + 1, 0)

# this gives the block bandwidth in each block column/row
@inline colblockbandwidths(A::AbstractMatrix) = Fill.(blockbandwidths(A), nblocks(A,2))
@inline rowblockbandwidths(A::AbstractMatrix) = Fill.(blockbandwidths(A), nblocks(A,1))

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


## BlockSlice1 is a conveneience for views

const BlockSlice1 = BlockSlice{Block{1,Int}}


######################################
#  RaggedMatrix interface
######################################



@inline function colstart(A::AbstractBlockBandedMatrix, i::Integer)
    bs = A.block_sizes.block_sizes
    bs.cumul_sizes[1][Int(blockcolstart(A, _find_block(bs, 2, i)[1]))]
end
@inline function colstop(A::AbstractBlockBandedMatrix, i::Integer)
    bs = A.block_sizes.block_sizes
    bs.cumul_sizes[1][Int(blockcolstop(A, _find_block(bs, 2, i)[1]))+1]-1
end
@inline function rowstart(A::AbstractBlockBandedMatrix, i::Integer)
    bs = A.block_sizes.block_sizes
    bs.cumul_sizes[2][Int(blockrowstart(A, _find_block(bs, 1, i)[1]))]
end
@inline function rowstop(A::AbstractBlockBandedMatrix, i::Integer)
    bs = A.block_sizes.block_sizes
    bs.cumul_sizes[2][Int(blockrowstop(A, _find_block(bs, 1, i)[1]))+1]-1
end

# default implementation loops over all indices, including zeros
function fill!(A::AbstractBlockBandedMatrix, val::Any)
  iszero(val) || throw(BandError(A))
  fill!(A.data, val)
  A
end
