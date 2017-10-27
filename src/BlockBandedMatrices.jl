__precompile__()

module BlockBandedMatrices
using BlockArrays, BandedMatrices

import BlockArrays: BlockSizes, nblocks, blocksize, blockcheckbounds, global2blockindex,
                        Block, BlockSlice, getblock

import BandedMatrices: isbanded, leadingdimension, bandwidth, banded_getindex,
                        inbands_setindex!, inbands_getindex, banded_setindex!

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert

export BandedBlockBandedMatrix

# A BlockBandedMatrix is a BlockMatrix, but is not a BandedMatrix
abstract type AbstractBlockBandedMatrix{T} <: AbstractBlockMatrix{T} end

include("BandedBlockBandedMatrix.jl")
end # module
