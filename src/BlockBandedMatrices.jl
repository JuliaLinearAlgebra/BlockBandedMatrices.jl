__precompile__()

module BlockBandedMatrices
using BlockArrays, BandedMatrices

import BlockArrays: BlockSizes, nblocks, blocksize, blockcheckbounds, global2blockindex,
                        Block, BlockSlice, getblock

import BandedMatrices: isbanded, leadingdimension, bandwidth, banded_getindex,
                        inbands_setindex!, inbands_getindex, banded_setindex!,
                        banded_generic_axpy!, banded_A_mul_B!, BLASBandedMatrix,
                        BlasFloat, banded_dense_axpy!, banded_A_mul_B

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        isdiag, +, *, -, /, \

export BandedBlockBandedMatrix

# A BlockBandedMatrix is a BlockMatrix, but is not a BandedMatrix
abstract type AbstractBlockBandedMatrix{T} <: AbstractBlockMatrix{T} end

include("BandedBlockBandedMatrix.jl")
end # module
