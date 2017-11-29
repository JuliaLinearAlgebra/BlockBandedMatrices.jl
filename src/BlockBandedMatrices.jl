__precompile__()

module BlockBandedMatrices
using BlockArrays, BandedMatrices

import BlockArrays: BlockSizes, nblocks, blocksize, blockcheckbounds, global2blockindex,
                        Block, BlockSlice, getblock, unblock

import BandedMatrices: isbanded, leadingdimension, bandwidth, banded_getindex,
                        inbands_setindex!, inbands_getindex, banded_setindex!,
                        banded_generic_axpy!, banded_A_mul_B!,
                        BlasFloat, banded_dense_axpy!, blasstructure,
                        BandedSubBandedMatrix, αA_mul_B_plus_βC!,
                        @banded_banded_linalg, @banded_linalg, @banded,
                        _BandedMatrix

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        isdiag, +, *, -, /, \, strides, zeros, eye

export BandedBlockBandedMatrix, BlockBandedMatrix, blockbandwidth, blockbandwidths,
        subblockbandwidth, subblockbandwidths

include("AbstractBlockBandedMatrix.jl")
include("BlockBandedMatrix.jl")
include("BandedBlockBandedMatrix.jl")
include("linalg.jl")

end # module
