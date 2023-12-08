module BlockBandedMatrices
using BlockArrays, BandedMatrices, ArrayLayouts, FillArrays, MatrixFactorizations
using LinearAlgebra

import Base: getindex, setindex!, checkbounds, @propagate_inbounds, convert,
                        +, *, -, /, \, strides, zeros, size,
                        unsafe_convert, fill!, length, first, last,
                        eltype, getindex, to_indices, to_index,
                        reindex, tail, @_propagate_inbounds_meta,
                        ==, axes, copy, copyto!, similar, OneTo, Slice

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted, broadcasted

import LinearAlgebra: UniformScaling, isdiag, rmul!, lmul!, ldiv!, rdiv!, axpy!,
                        AbstractTriangular, AdjOrTrans, HermOrSym, StructuredMatrixStyle,
                        qr, qr!
import LinearAlgebra.BLAS: BlasInt, BlasFloat, @blasfunc, BlasComplex, BlasReal
import LinearAlgebra.LAPACK: chktrans, chkdiag, chklapackerror, checksquare, chkstride1,
                    chkuplo
import MatrixFactorizations: ql, ql!, _ql, QLPackedQ, AdjQRPackedQLayout, AdjQLPackedQLayout, QR, QRPackedQ

import ArrayLayouts: BlasMatLmulVec, MatLmulVec, MatLmulMat,
                    triangularlayout, UpperTriangularLayout, TriangularLayout, MatLdivVec,
                    triangulardata, sublayout, sub_materialize, materialize, materialize!,
                    AbstractColumnMajor, DenseColumnMajor, ColumnMajor,
                    DiagonalLayout, MulAdd, mul, colsupport, rowsupport,
                    _qr, _factorize, _copyto!, zero!, layout_replace_in_print_matrix,
                    transposelayout, conjlayout, symmetriclayout, hermitianlayout

import BlockArrays: blocksize, blockcheckbounds, BlockedUnitRange, blockisequal, DefaultBlockAxis,
                        Block, BlockSlice, unblock, block, blockindex,
                        _blocklengths2blocklasts, BlockIndexRange, sizes_from_blocks, BlockSlice1,
                        blockcolsupport, blockrowsupport, blockcolstart, blockcolstop, blockrowstart, blockrowstop,
                        AbstractBlockLayout, BlockLayout, blocks, hasmatchingblocks, BlockStyle, BlockSlices, _blockkron

import BandedMatrices: isbanded, bandwidths, bandwidth, banded_getindex, colrange,
                        inbands_setindex!, inbands_getindex, banded_setindex!,
                        banded_generic_axpy!,
                        BlasFloat, banded_dense_axpy!, MemoryLayout,
                        BandedLayout, BandedColumnMajor, BandedColumns, bandedcolumns,
                        BandedSubBandedMatrix, bandeddata,
                        _BandedMatrix, colstart, colstop, rowstart, rowstop,
                        BandedStyle, bandshift

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

if !isdefined(Base, :get_extension)
    include("../ext/BlockBandedMatricesSparseArraysExt.jl")
end

end # module
