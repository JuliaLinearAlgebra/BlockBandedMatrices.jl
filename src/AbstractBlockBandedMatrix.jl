# AbstractBandedMatrix must implement

# A BlockBandedMatrix is a BlockMatrix, but is not a BandedMatrix
abstract type AbstractBlockBandedMatrix{T} <: AbstractBlockMatrix{T} end

doc"""
    blockbandwidths(A)

Returns a tuple containing the upper and lower blockbandwidth of `A`.
"""
blockbandwidths(A::AbstractMatrix) = blockbandwidth(A,1),blockbandwidth(A,2)

doc"""
    blockbandwidth(A,i)

Returns the lower blockbandwidth (`i==1`) or the upper blockbandwidth (`i==2`).
"""
blockbandwidth(A::AbstractMatrix, k::Integer) = k==1 ? nblocks(A,1)-1 : nblocks(A,2)-1

doc"""
    bandrange(A)

Returns the range `-blockbandwidth(A,1):blockbandwidth(A,2)`.
"""
blockbandrange(A::AbstractMatrix) = -blockbandwidth(A,1):blockbandwidth(A,2)



# start/stop indices of the i-th column/row, bounded by actual matrix size
@inline blockcolstart(A::AbstractVecOrMat, i::Integer) = max(i-blockbandwidth(A,2), 1)
@inline  blockcolstop(A::AbstractVecOrMat, i::Integer) = max(min(i+blockbandwidth(A,1), nblocks(A, 1)), 0)
@inline blockrowstart(A::AbstractVecOrMat, i::Integer) = max(i-blockbandwidth(A,1), 1)
@inline  blockrowstop(A::AbstractVecOrMat, i::Integer) = max(min(i+blockbandwidth(A,2), nblocks(A, 2)), 0)


@inline blockcolrange(A::AbstractVecOrMat, i::Integer) = blockcolstart(A,i):blockcolstop(A,i)
@inline blockrowrange(A::AbstractVecOrMat, i::Integer) = blockrowstart(A,i):blockrowstop(A,i)


# length of i-the column/row
@inline blockcollength(A::AbstractVecOrMat, i::Integer) = max(blockcolstop(A, i) - blockcolstart(A, i) + 1, 0)
@inline blockrowlength(A::AbstractVecOrMat, i::Integer) = max(blockrowstop(A, i) - blockrowstart(A, i) + 1, 0)


doc"""
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
