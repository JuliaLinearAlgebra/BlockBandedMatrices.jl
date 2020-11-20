module BlockBandedMatrices
using BlockArrays, BandedMatrices, ArrayLayouts, FillArrays, SparseArrays, MatrixFactorizations
using LinearAlgebra

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        +, *, -, /, \, strides, zeros, size,
                        unsafe_convert, fill!, length, first, last,
                        eltype, getindex, to_indices, to_index,
                        reindex, _maybetail, tail, @_propagate_inbounds_meta,
                        ==, axes, copyto!, similar, OneTo

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted, broadcasted,
                        materialize, materialize!

import LinearAlgebra: UniformScaling, isdiag, rmul!, lmul!, ldiv!, rdiv!,
                        AbstractTriangular, AdjOrTrans, HermOrSym, StructuredMatrixStyle,
                        qr, qr!
import LinearAlgebra.BLAS: BlasInt, BlasFloat, @blasfunc, libblas, BlasComplex, BlasReal
import LinearAlgebra.LAPACK: chktrans, chkdiag, liblapack, chklapackerror, checksquare, chkstride1,
                    chkuplo
import MatrixFactorizations: ql, ql!, _ql, QLPackedQ, AdjQRPackedQLayout, AdjQLPackedQLayout, QR, QRPackedQ
import SparseArrays: sparse

import ArrayLayouts: BlasMatLmulVec, MatLmulVec, MatLmulMat,
                    triangularlayout, UpperTriangularLayout, TriangularLayout, MatLdivVec,
                    triangulardata, sublayout, sub_materialize,
                    AbstractColumnMajor, DenseColumnMajor, ColumnMajor,
                    DiagonalLayout, MulAdd, mul, colsupport, rowsupport,
                    _qr, _factorize, _copyto!, zero!, layout_replace_in_print_matrix

import BlockArrays: blocksize, blockcheckbounds, BlockedUnitRange, blockisequal, DefaultBlockAxis,
                        Block, BlockSlice, getblock, unblock, setblock!, block, blockindex,
                        _blocklengths2blocklasts, BlockIndexRange, sizes_from_blocks, BlockSlice1,
                        blockcolsupport, blockrowsupport, blockcolstart, blockcolstop, blockrowstart, blockrowstop,
                        AbstractBlockLayout, BlockLayout, blocks, hasmatchingblocks, BlockStyle

import BandedMatrices: isbanded, bandwidths, bandwidth, banded_getindex, colrange,
                        inbands_setindex!, inbands_getindex, banded_setindex!,
                        banded_generic_axpy!,
                        BlasFloat, banded_dense_axpy!, MemoryLayout,
                        BandedLayout, BandedColumnMajor, BandedColumns,
                        BandedSubBandedMatrix, bandeddata,
                        _BandedMatrix, colstart, colstop, rowstart, rowstop,
                        BandedStyle, _fill_lmul!, bandshift,
                        _banded_colval, _banded_rowval, _banded_nzval # for sparse

export BandedBlockBandedMatrix, BlockBandedMatrix, BlockSkylineMatrix, blockbandwidth, blockbandwidths,
        subblockbandwidth, subblockbandwidths, Ones, Zeros, Fill, Block, BlockTridiagonal, BlockBidiagonal, isblockbanded


const Block1 = Block{1,Int}
const BlockRange1{R<:AbstractUnitRange{Int}} = BlockRange{1,Tuple{R}}
const BlockIndexRange1{R<:AbstractUnitRange{Int}} = BlockIndexRange{1,Tuple{R}}

blockcolrange(A...) = blockcolsupport(A...)
blockrowrange(A...) = blockrowsupport(A...)

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
