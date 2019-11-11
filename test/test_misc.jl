using ArrayLayouts, BlockBandedMatrices, BandedMatrices, BlockArrays, LinearAlgebra, Test
import Base: getindex, size
import BandedMatrices: bandwidths, AbstractBandedMatrix, BandedStyle, bandeddata, BandedColumns
import BlockArrays: blocksizes, BlockSizes


@testset "Diagonal interface" begin
    n = 10
    D = Diagonal(randn(n^2))
    @test blocksizes(D) == BlockSizes([n^2], [n^2])
    @test blockbandwidths(D) == subblockbandwidths(D) == (0,0)

    PD = PseudoBlockArray(D, Fill(n,n), Fill(n,n))
    @test blockbandwidths(PD) == bandwidths(PD) == (0,0)
    @test MemoryLayout(typeof(PD)) isa DiagonalLayout{DenseColumnMajor}
    @test bandeddata(PD) == bandeddata(D)

    V = view(PD, Block(1,1))
    @test bandwidths(V) == (0,0)
    @test Broadcast.BroadcastStyle(typeof(V)) == BandedStyle()
    @test MemoryLayout(typeof(V)) isa BandedColumns{DenseColumnMajor}
    @test bandeddata(V) == bandeddata(PD)[:,1:n]
    @test BandedMatrix(V) == V
end

@testset "Block Tridiagonal" begin
    A = BlockTridiagonal(fill([1 2],3), fill([3 4],4), fill([4 5],3))
    @test blockbandwidths(A) == (1,1)
    @test isblockbanded(A)
    @test A[Block(1,1)] == [3 4]
    @test @inferred(A[Block(1,2)]) == [4 5]
    @test @inferred(getblock(A,1,3)) == @inferred(A[Block(1,3)]) == [0 0]
    @test_throws DimensionMismatch A+I
    A = BlockTridiagonal(fill([1 2; 1 2],3), fill([3 4; 3 4],4), fill([4 5; 4 5],3))
    @test A+I == I+A == mortar(Tridiagonal(fill([1 2; 1 2],3), fill([4 4; 3 5],4), fill([4 5; 4 5],3))) == Matrix(A) + I
end


