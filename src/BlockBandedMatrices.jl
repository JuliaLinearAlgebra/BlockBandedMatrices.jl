__precompile__()

module BlockBandedMatrices
using BlockArrays, BandedMatrices, LazyArrays, FillArrays
using LinearAlgebra

import BlockArrays: BlockSizes, nblocks, blocksize, blockcheckbounds, global2blockindex,
                        Block, BlockSlice, getblock, unblock, setblock!, globalrange,
                        _unblock, _find_block, BlockIndexRange

import BandedMatrices: isbanded, leadingdimension, bandwidths, bandwidth, banded_getindex,
                        inbands_setindex!, inbands_getindex, banded_setindex!,
                        banded_generic_axpy!,
                        BlasFloat, banded_dense_axpy!, MemoryLayout,
                        BandedColumnMajor,
                        BandedSubBandedMatrix, bandeddata,
                        _BandedMatrix, colstart, colstop, rowstart, rowstop

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        +, *, -, /, \, strides, zeros, size,
                        unsafe_convert, fill!, length, first, last,
                        eltype, getindex, to_indices, to_index,
                        reindex, _maybetail, tail, @_propagate_inbounds_meta

import LinearAlgebra: UniformScaling, isdiag
import LinearAlgebra.BLAS: BlasInt, BlasFloat, @blasfunc, libblas, BlasComplex
import LinearAlgebra.LAPACK: chktrans, chkdiag, liblapack, chklapackerror, checksquare, chkstride1,
                    chkuplo


import Compat: axes, copyto!

import LinearAlgebra: rmul!, lmul!, ldiv!, rdiv!

import LazyArrays: AbstractStridedLayout, ColumnMajor, @blasmatvec

export BandedBlockBandedMatrix, BlockBandedMatrix, blockbandwidth, blockbandwidths,
        subblockbandwidth, subblockbandwidths, Ones, Zeros, Fill, Block


include("AbstractBlockBandedMatrix.jl")
include("BlockBandedMatrix.jl")
include("BandedBlockBandedMatrix.jl")

include("linalg.jl")

end # module
