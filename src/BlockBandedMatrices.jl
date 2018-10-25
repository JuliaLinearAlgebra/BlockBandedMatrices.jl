
module BlockBandedMatrices
using BlockArrays, BandedMatrices, LazyArrays, FillArrays, SparseArrays
using LinearAlgebra

import BlockArrays: BlockSizes, nblocks, blocksize, blockcheckbounds, global2blockindex,
                        Block, BlockSlice, getblock, unblock, setblock!, globalrange,
                        _unblock, _find_block, BlockIndexRange, blocksizes, cumulsizes,
                        AbstractBlockSizes

import BandedMatrices: isbanded, bandwidths, bandwidth, banded_getindex, colrange,
                        inbands_setindex!, inbands_getindex, banded_setindex!,
                        banded_generic_axpy!,
                        BlasFloat, banded_dense_axpy!, MemoryLayout,
                        BandedColumnMajor,
                        BandedSubBandedMatrix, bandeddata, tribandeddata,
                        _BandedMatrix, colstart, colstop, rowstart, rowstop,
                        BandedStyle,
                        _banded_colval, _banded_rowval, _banded_nzval # for sparse

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        +, *, -, /, \, strides, zeros, size,
                        unsafe_convert, fill!, length, first, last,
                        eltype, getindex, to_indices, to_index,
                        reindex, _maybetail, tail, @_propagate_inbounds_meta,
                        ==, axes, copyto!, similar

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted, broadcasted

import LinearAlgebra: UniformScaling, isdiag, rmul!, lmul!, ldiv!, rdiv!,
                        AbstractTriangular, AdjOrTrans, HermOrSym
import LinearAlgebra.BLAS: BlasInt, BlasFloat, @blasfunc, libblas, BlasComplex
import LinearAlgebra.LAPACK: chktrans, chkdiag, liblapack, chklapackerror, checksquare, chkstride1,
                    chkuplo

import SparseArrays: sparse

import LazyArrays: AbstractStridedLayout, ColumnMajor, @blasmatvec, @blasmatmat, @lazymul, blasmul!,
                    triangularlayout, UpperTriangularLayout, TriangularLayout, MatMulVec, MatLdivVec,
                    triangulardata, subarraylayout, _copyto!, @lazyldiv, @lazylmul,
                    ArrayMulArrayStyle

export BandedBlockBandedMatrix, BlockBandedMatrix, BlockTridiagonalMatrix, RaggedBlockBandedMatrix, blockbandwidth, blockbandwidths,
        subblockbandwidth, subblockbandwidths, Ones, Zeros, Fill, Block


include("AbstractBlockBandedMatrix.jl")
include("broadcast.jl")
include("BlockBandedMatrix.jl")
include("BandedBlockBandedMatrix.jl")

include("linalg.jl")

include("interfaceimpl.jl")
include("triblockbanded.jl")
include("adjtransblockbanded.jl")

end # module
