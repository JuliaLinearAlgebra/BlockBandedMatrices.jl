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
                        @banded_banded_linalg, @banded_linalg, @banded

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        isdiag, +, *, -, /, \, strides

export BandedBlockBandedMatrix, BlockBandedMatrix, blockbandwidth, blockbandwidths,
        subblockbandwidth, subblockbandwidths

include("AbstractBlockBandedMatrix.jl")
include("BlockBandedMatrix.jl")
include("BandedBlockBandedMatrix.jl")

# function Base.A_mul_B!(Y::AbstractBlockBandedMatrix, A::AbstractBlockBandedMatrix, B::AbstractBlockBandedMatrix)
#     T=eltype(Y)
#     BLAS.scal!(length(Y.data),zero(T),Y.data,1)
#     o=one(T)
#     for J=Block(1):Block(blocksize(B,2)),
#         N=blockcolrange(B,J), K=blockcolrange(A,N)
#         αA_mul_B_plus_βC!(o,view(A,K,N),view(B,N,J),o,view(Y,K,J))
#     end
#     Y
# end
end # module
