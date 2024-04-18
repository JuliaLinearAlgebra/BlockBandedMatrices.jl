module BlockBandedMatrices
using BlockArrays, BandedMatrices, ArrayLayouts, FillArrays, MatrixFactorizations
using LinearAlgebra

import ArrayLayouts: AbstractColumnMajor, AdjQRPackedQLayout, BlasMatLmulVec, ColumnMajor, DenseColumnMajor,
                     DiagonalLayout, MatLdivVec, MatLmulMat, MatLmulVec, MemoryLayout, MulAdd, TriangularLayout,
                     UpperTriangularLayout, _copyto!, _factorize, _qr, colsupport, conjlayout, hermitianlayout,
                     layout_replace_in_print_matrix, mul, rowsupport, sub_materialize, sublayout, symmetriclayout,
                     transposelayout, triangulardata, triangularlayout, zero!, materialize!, materialize

import BandedMatrices: BandedColumnMajor, BandedColumns, BandedLayout, BandedStyle, BandedSubBandedMatrix, BlasFloat,
                       _BandedMatrix, banded_dense_axpy!, banded_generic_axpy!, banded_getindex, banded_setindex!,
                       bandedcolumns, bandeddata, bandshift, bandwidth, bandwidths, colrange, colstart, colstop,
                       inbands_getindex, inbands_setindex!, isbanded, rowstart, rowstop

import Base: *, +, -, /, \, ==, @propagate_inbounds, OneTo, Slice, axes, checkbounds,
             convert, copy, copyto!, eltype, fill!, first, getindex, last, length, reindex, setindex!, similar, size,
             strides, unsafe_convert, zeros

import Base.Broadcast: AbstractArrayStyle, BroadcastStyle, Broadcasted, DefaultArrayStyle, broadcasted

import BlockArrays: AbstractBlockLayout, Block, BlockIndexRange, BlockLayout, BlockSlice, BlockSlice1, BlockSlices,
                    BlockStyle, BlockedUnitRange, DefaultBlockAxis, _blockkron, _blocklengths2blocklasts, block,
                    blockcheckbounds, blockcolstart, blockcolstop, blockcolsupport, blockindex, blockisequal,
                    blockrowstart, blockrowstop, blockrowsupport, blocks, blocksize, hasmatchingblocks,
                    sizes_from_blocks, unblock

import FillArrays: Fill, Ones, Zeros

import LinearAlgebra: AbstractTriangular, AdjOrTrans, HermOrSym, StructuredMatrixStyle, UniformScaling, axpy!,
                      isdiag, ldiv!, lmul!, qr, qr!, rdiv!, rmul!

import LinearAlgebra.BLAS: BlasComplex, BlasFloat, BlasReal

import MatrixFactorizations: AdjQLPackedQLayout, QR, QRPackedQ, _ql, ql, ql!


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
