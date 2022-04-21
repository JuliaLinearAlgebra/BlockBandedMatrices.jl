using ArrayLayouts, BlockBandedMatrices, BandedMatrices, BlockArrays, LinearAlgebra, Test
import BlockBandedMatrices: AbstractBandedBlockBandedMatrix, AbstractBlockBandedMatrix, block, blockindex, blockcolsupport, blockrowsupport
import BandedMatrices: bandwidths, AbstractBandedMatrix, BandedStyle, bandeddata, BandedColumns, _BandedMatrix

struct MyBandedBlockBandedMatrix <: AbstractBandedBlockBandedMatrix{Float64}
    A::BlockMatrix{Float64}
end

BlockBandedMatrices.blockbandwidths(::MyBandedBlockBandedMatrix) = (1,1)
BlockBandedMatrices.subblockbandwidths(::MyBandedBlockBandedMatrix) = (1,1)
Base.axes(A::MyBandedBlockBandedMatrix) = axes(A.A)
function Base.getindex(A::MyBandedBlockBandedMatrix, k::Int, j::Int)
    Kk, Jj = findblockindex(axes(A,1), k), findblockindex(axes(A,2), j)
    -1 ≤ Int(block(Kk)-block(Jj)) ≤ 1 || return 0.0
    -1 ≤ Int(blockindex(Kk)-blockindex(Jj)) ≤ 1 || return 0.0
    A.A[k,j]
end


struct MyBlockBandedMatrix <: AbstractBlockBandedMatrix{Float64}
    A::BlockMatrix{Float64}
end

BlockBandedMatrices.blockbandwidths(::MyBlockBandedMatrix) = (1,1)
Base.axes(A::MyBlockBandedMatrix) = axes(A.A)
function Base.getindex(A::MyBlockBandedMatrix, k::Int, j::Int)
    Kk, Jj = findblockindex(axes(A,1), k), findblockindex(axes(A,2), j)
    -1 ≤ Int(block(Kk)-block(Jj)) ≤ 1 || return 0.0
    A.A[k,j]
end


@testset "Interfaces" begin
    @testset "MyBandedBlockBandedMatrix" begin
        A = MyBandedBlockBandedMatrix(BlockMatrix(randn(6,6), 1:3, 1:3))
        @test MemoryLayout(A) isa BlockBandedMatrices.BandedBlockBandedLayout
        @test A[Block(3,3)] == BandedMatrix(A.A[Block(3,3)],(1,1))
        @test A[Block.(1:3),Block.(1:3)] == A

        @test A[Block(3,3)] isa BandedMatrix
        @test A[Block(3)[2:3],Block(3)[1:2]] isa BandedMatrix
        @test A[Block(3)[2:3],Block(3)] isa BandedMatrix
        @test A[Block(3),Block(3)[1:2]] isa BandedMatrix
        @test A[Block.(1:3),Block.(1:3)] isa BandedBlockBandedMatrix
        @test A[Block(1),Block.(2:3)] isa PseudoBlockArray
        @test A[Block.(2:3),Block(1)] isa PseudoBlockArray
        @test A[Block.(2:3),Block(2)[1:2]] isa PseudoBlockArray
        @test A[Block(2)[1:2],Block.(2:3)] isa PseudoBlockArray
    end

    @testset "MyBlockBandedMatrix" begin
        A = MyBlockBandedMatrix(BlockMatrix(randn(6,6), 1:3, 1:3))
        @test MemoryLayout(A) isa BlockBandedMatrices.BlockBandedLayout
        @test MemoryLayout(A') isa BlockBandedMatrices.BlockBandedLayout
        @test MemoryLayout(transpose(A)) isa BlockBandedMatrices.BlockBandedLayout
        @test A[Block(3,3)] == A.A[Block(3,3)]
        @test A[Block.(1:3),Block.(1:3)] == A

        @test A[Block(3,3)] isa Matrix
        @test A[Block(3)[2:3],Block(3)[1:2]] isa Matrix
        @test A[Block(3)[2:3],Block(3)] isa Matrix
        @test A[Block(3),Block(3)[1:2]] isa Matrix
        @test A[Block.(1:3),Block.(1:3)] isa BlockBandedMatrix
        @test A[Block(1),Block.(2:3)] isa PseudoBlockArray
        @test A[Block.(2:3),Block(1)] isa PseudoBlockArray
        @test A[Block.(2:3),Block(2)[1:2]] isa PseudoBlockArray
        @test A[Block(2)[1:2],Block.(2:3)] isa PseudoBlockArray
    end

    @testset "Zeros" begin
        Z = Zeros(blockedrange(1:3), blockedrange(1:3))
        B = BandedBlockBandedMatrix(Z)
        @test B == Z
        @test blockisequal(axes(Z), axes(B))
        @test blockbandwidths(B) == blockbandwidths(Z) == (-1,-1)
        @test subblockbandwidths(B) == subblockbandwidths(Z) == (-1,-1)
    end

    @testset "Diagonal interface" begin
        n = 10
        D = Diagonal(randn(n^2))
        @test blockbandwidths(D) == subblockbandwidths(D) == (0,0)

        PD = PseudoBlockArray(D, Fill(n,n), Fill(n,n))
        @test blockbandwidths(PD) == bandwidths(PD) == (0,0)
        @test MemoryLayout(typeof(PD)) isa DiagonalLayout{DenseColumnMajor}
        @test bandeddata(PD) == bandeddata(D)
        V = view(PD, Block(1,1))

        @test bandwidths(V) == (0,0)
        @test_broken Broadcast.BroadcastStyle(typeof(V)) == BandedStyle()
        @test MemoryLayout(typeof(V)) isa BandedColumns{DenseColumnMajor}
        @test bandeddata(V) == bandeddata(PD)[:,1:n]
        @test BandedMatrix(V) == V
    end

    @testset "Block Diagonal" begin
        A = BlockBandedMatrices.BlockDiagonal(fill([1 2],3))
        @test blockbandwidths(A) == (0,0)
        @test isblockbanded(A)
        @test A[Block(1,1)] == [1 2]
        @test @inferred(A[Block(1,2)]) == [0 0]
        @test_throws DimensionMismatch A+I
        A = BlockBandedMatrices.BlockDiagonal(fill([1 2; 1 2],3))
        @test A+I == I+A == mortar(Diagonal(fill([2 2; 1 3],3))) == Matrix(A) + I
    end

    @testset "Block Bidiagonal" begin
        Bu = BlockBidiagonal(fill([1 2],4), fill([3 4],3), :U)
        Bl = BlockBidiagonal(fill([1 2],4), fill([3 4],3), :L)
        @test blockbandwidths(Bu) == (0,1)
        @test blockbandwidths(Bl) == (1,0)
        @test isblockbanded(Bu)
        @test isblockbanded(Bl)
        @test Bu[Block(1,1)] == Bl[Block(1,1)] == [1 2]
        @test @inferred(Bu[Block(1,2)]) == @inferred(Bl[Block(2,1)]) == [3 4]
        @test @inferred(view(Bu,Block(1,3))) == @inferred(Bu[Block(1,3)]) == [0 0]
        @test_throws DimensionMismatch Bu+I
        Bu = BlockBidiagonal(fill([1 2; 1 2],4), fill([3 4; 3 4],3), :U)
        Bl = BlockBidiagonal(fill([1 2; 1 2],4), fill([3 4; 3 4],3), :L)
        @test Bu+I == I+Bu == mortar(Bidiagonal(fill([2 2; 1 3],4), fill([3 4; 3 4],3), :U)) == Matrix(Bu) + I
        @test Bl+I == I+Bl == mortar(Bidiagonal(fill([2 2; 1 3],4), fill([3 4; 3 4],3), :L)) == Matrix(Bl) + I
        @test Bu-I == mortar(Bidiagonal(fill([0 2; 1 1],4), fill([3 4; 3 4],3), :U)) == Matrix(Bu) - I
        @test I-Bu == mortar(Bidiagonal(fill([0 -2; -1 -1],4), fill(-[3 4; 3 4],3), :U)) == I - Matrix(Bu)
    end

    @testset "Block Tridiagonal" begin
        A = BlockTridiagonal(fill([1 2],3), fill([3 4],4), fill([4 5],3))
        @test blockbandwidths(A) == (1,1)
        @test isblockbanded(A)
        @test A[Block(1,1)] == [3 4]
        @test @inferred(A[Block(1,2)]) == [4 5]
        @test @inferred(view(A,Block(1,3))) == @inferred(A[Block(1,3)]) == [0 0]
        @test_throws DimensionMismatch A+I
        A = BlockTridiagonal(fill([1 2; 1 2],3), fill([3 4; 3 4],4), fill([4 5; 4 5],3))
        @test A+I == I+A == mortar(Tridiagonal(fill([1 2; 1 2],3), fill([4 4; 3 5],4), fill([4 5; 4 5],3))) == Matrix(A) + I
        @test A-I == mortar(Tridiagonal(fill([1 2; 1 2],3), fill([2 4; 3 3],4), fill([4 5; 4 5],3))) == Matrix(A) - I
        @test I-A == mortar(Tridiagonal(fill(-[1 2; 1 2],3), fill([-2 -4; -3 -3],4), fill(-[4 5; 4 5],3))) == I - Matrix(A)
    end

    @testset "DiagonalBlock" begin
        D = Diagonal(PseudoBlockVector(randn(6), 1:3))
        D̃ = BandedBlockBandedMatrix(D)
        @test D̃ == D
        @test blockbandwidths(D) == blockbandwidths(D̃) == subblockbandwidths(D) == subblockbandwidths(D̃) == (0,0)

        @test @inferred(D[Block(2,2)]) isa BandedMatrix
        @test D[Block(2,3)] isa BandedMatrix
        @test bandwidths(D[Block(2,2)]) == (0,0)
        @test bandwidths(D[Block(2,3)]) == (-720,-720)
        @test D[Block(2,2)] == D[2:3,2:3]
        @test D[Block(2,3)] == zeros(2,3)

        @test D[Block.(1:2),Block.(2:3)] isa BandedBlockBandedMatrix
        @test D[Block.(1:2),Block.(2:3)] == D[1:3,2:6]
        @test blockbandwidths(D[Block.(1:2),Block.(2:3)]) == (1,-1)
        @test subblockbandwidths(D[Block.(1:2),Block.(2:3)]) == (0,0)
    end

    @testset "Block-BandedMatrix" begin
        a = blockedrange(1:5)
        B = _BandedMatrix(PseudoBlockArray(randn(5,length(a)),(Base.OneTo(5),a)), a, 3, 1)
        @test blockcolsupport(B,Block(1)) == Block.(1:3)
        @test blockcolsupport(B,Block(3)) == Block.(2:4)
        @test blockrowsupport(B,Block(1)) == Block.(1:2)
        @test blockrowsupport(B,Block(4)) == Block.(3:5)

        Q = Eye((a,))[:,Block(2)]
        @test Q isa BandedMatrix
        @test blockcolsupport(Q,Block(1)) == Block.(2:2)

        Q = Eye((a,))[Block(2),:]
        @test Q isa BandedMatrix
        @test blockrowsupport(Q,Block(1)) == Block.(2:2)

        @testset "constant blocks" begin
            a = blockedrange(Fill(2,5))
            Q = Eye((a,))[:,Block(2)]
            @test blockbandwidths(Q) == (1,-1)

            B = _BandedMatrix(randn(5,length(a)), a, 3, 1)
            @test blockbandwidths(B) == (4,0)
        end
    end
end
