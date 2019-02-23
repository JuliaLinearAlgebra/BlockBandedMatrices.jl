
module BlockBandedMatrices
using BlockArrays, BandedMatrices, LazyArrays, FillArrays, SparseArrays
using LinearAlgebra

import BlockArrays: BlockSizes, nblocks, blocksize, blockcheckbounds, global2blockindex,
                        Block, BlockSlice, getblock, unblock, setblock!, globalrange,
                        _unblock, _find_block, BlockIndexRange, blocksizes, cumulsizes,
                        AbstractBlockSizes, sizes_from_blocks

import BandedMatrices: isbanded, bandwidths, bandwidth, banded_getindex, colrange,
                        inbands_setindex!, inbands_getindex, banded_setindex!,
                        banded_generic_axpy!,
                        BlasFloat, banded_dense_axpy!, MemoryLayout,
                        BandedColumnMajor,
                        BandedSubBandedMatrix, bandeddata, tribandeddata,
                        _BandedMatrix, colstart, colstop, rowstart, rowstop,
                        BandedStyle, _fill_lmul!,
                        _banded_colval, _banded_rowval, _banded_nzval # for sparse

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        +, *, -, /, \, strides, zeros, size,
                        unsafe_convert, fill!, length, first, last,
                        eltype, getindex, to_indices, to_index,
                        reindex, _maybetail, tail, @_propagate_inbounds_meta,
                        ==, axes, copyto!, similar, OneTo

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted, broadcasted,
                        materialize, materialize!

import LinearAlgebra: UniformScaling, isdiag, rmul!, lmul!, ldiv!, rdiv!,
                        AbstractTriangular, AdjOrTrans, HermOrSym, StructuredMatrixStyle
import LinearAlgebra.BLAS: BlasInt, BlasFloat, @blasfunc, libblas, BlasComplex
import LinearAlgebra.LAPACK: chktrans, chkdiag, liblapack, chklapackerror, checksquare, chkstride1,
                    chkuplo

import SparseArrays: sparse

import LazyArrays: AbstractStridedLayout, ColumnMajor, @lazymul, MatMulMatAdd, MatMulVecAdd,
                    triangularlayout, UpperTriangularLayout, TriangularLayout, MatMulVec, MatLdivVec,
                    triangulardata, subarraylayout, _copyto!, @lazyldiv, @lazylmul,
                    ArrayMulArrayStyle, AbstractColumnMajor, DenseColumnMajor, ColumnMajor,
                    DiagonalLayout, MatMulMat

export BandedBlockBandedMatrix, BlockBandedMatrix, BlockTridiagonalMatrix, BlockSkylineMatrix, blockbandwidth, blockbandwidths,
        subblockbandwidth, subblockbandwidths, Ones, Zeros, Fill, Block


include("AbstractBlockBandedMatrix.jl")
include("broadcast.jl")
include("BlockSkylineMatrix.jl")
include("BandedBlockBandedMatrix.jl")

include("linalg.jl")

include("interfaceimpl.jl")
include("triblockbanded.jl")
include("adjtransblockbanded.jl")

end # module
