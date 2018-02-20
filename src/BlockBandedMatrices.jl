__precompile__()

module BlockBandedMatrices
using BlockArrays, BandedMatrices, FillArrays, Compat

import BlockArrays: BlockSizes, nblocks, blocksize, blockcheckbounds, global2blockindex,
                        Block, BlockSlice, getblock, unblock, setblock!, globalrange,
                        _find_block

import BandedMatrices: isbanded, leadingdimension, bandwidth, banded_getindex,
                        inbands_setindex!, inbands_getindex, banded_setindex!,
                        banded_generic_axpy!, banded_A_mul_B!,
                        BlasFloat, banded_dense_axpy!, memorylayout,
                        BandedLayout, StridedLayout,
                        BandedSubBandedMatrix, mul!, _mul!,
                        @banded_banded_linalg, @banded_linalg, @banded,
                        _BandedMatrix, colstart, colstop, rowstart, rowstop

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        isdiag, +, *, -, /, \, strides, zeros, eye, size,
                        unsafe_convert, copy!, fill!

import Base.LinAlg: A_ldiv_B!, A_mul_B!
import Base.BLAS: BlasInt, BlasFloat, @blasfunc, libblas
import Base.LAPACK: chktrans, chkdiag, liblapack, chklapackerror, checksquare, chkstride1,
                    chkuplo

export BandedBlockBandedMatrix, BlockBandedMatrix, blockbandwidth, blockbandwidths,
        subblockbandwidth, subblockbandwidths

include("lapack.jl")

include("AbstractBlockBandedMatrix.jl")
include("BlockBandedMatrix.jl")
include("BandedBlockBandedMatrix.jl")

include("linalg.jl")

end # module
