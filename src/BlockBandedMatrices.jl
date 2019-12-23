module BlockBandedMatrices
using BlockArrays, BandedMatrices, ArrayLayouts, FillArrays, SparseArrays, MatrixFactorizations
using LinearAlgebra

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        +, *, -, /, \, strides, zeros, size,
                        unsafe_convert, fill!, length, first, last,
                        eltype, getindex, to_indices, to_index,
                        reindex, _maybetail, tail, @_propagate_inbounds_meta,
                        ==, axes, copyto!, similar, OneTo, replace_in_print_matrix

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted, broadcasted,
                        materialize, materialize!

import LinearAlgebra: UniformScaling, isdiag, rmul!, lmul!, ldiv!, rdiv!,
                        AbstractTriangular, AdjOrTrans, HermOrSym, StructuredMatrixStyle,
                        qr, qr!, QRPackedQ
import LinearAlgebra.BLAS: BlasInt, BlasFloat, @blasfunc, libblas, BlasComplex, BlasReal
import LinearAlgebra.LAPACK: chktrans, chkdiag, liblapack, chklapackerror, checksquare, chkstride1,
                    chkuplo
import MatrixFactorizations: ql, ql!, QLPackedQ
import SparseArrays: sparse

import ArrayLayouts: @lazymul, MatMulMatAdd, MatMulVecAdd, BlasMatLmulVec,
                    triangularlayout, UpperTriangularLayout, TriangularLayout, MatLdivVec,
                    triangulardata, sublayout, @lazyldiv, @lazylmul,
                    AbstractColumnMajor, DenseColumnMajor, ColumnMajor,
                    DiagonalLayout, materialize!, MulAdd, mul

import BlockArrays: blocksize, blockcheckbounds, BlockedUnitRange, blockisequal, DefaultBlockAxis,
                        Block, BlockSlice, getblock, unblock, setblock!, block, blockindex,
                        _blocklengths2blocklasts, BlockIndexRange, sizes_from_blocks

import BandedMatrices: isbanded, bandwidths, bandwidth, banded_getindex, colrange,
                        inbands_setindex!, inbands_getindex, banded_setindex!,
                        banded_generic_axpy!,
                        BlasFloat, banded_dense_axpy!, MemoryLayout,
                        BandedColumnMajor,
                        BandedSubBandedMatrix, bandeddata, tribandeddata,
                        _BandedMatrix, colstart, colstop, rowstart, rowstop,
                        BandedStyle, _fill_lmul!,
                        _banded_colval, _banded_rowval, _banded_nzval # for sparse

export BandedBlockBandedMatrix, BlockBandedMatrix, BlockSkylineMatrix, blockbandwidth, blockbandwidths,
        subblockbandwidth, subblockbandwidths, Ones, Zeros, Fill, Block, BlockTridiagonal, isblockbanded


include("AbstractBlockBandedMatrix.jl")
include("broadcast.jl")
include("BlockSkylineMatrix.jl")
include("BandedBlockBandedMatrix.jl")

include("linalg.jl")
include("blockskylineqr.jl")

include("interfaceimpl.jl")
include("triblockbanded.jl")
include("adjtransblockbanded.jl")

end # module
