using LazyArrays, BlockBandedMatrices, BandedMatrices, BlockArrays, LinearAlgebra, Test
import Base: getindex, size
import BandedMatrices: bandwidths, AbstractBandedMatrix, BandedStyle
import BlockArrays: blocksizes, BlockSizes

struct FiniteDifference{T} <: AbstractBandedMatrix{T}
    n::Int
end

FiniteDifference(n) = FiniteDifference{Float64}(n)

getindex(F::FiniteDifference{T}, k::Int, j::Int) where T =
    if k == j
        -2*one(T)*F.n^2
    elseif abs(k-j) == 1
        one(T)*F.n^2
    else
        zero(T)
    end

bandwidths(F::FiniteDifference) = (1,1)
size(F::FiniteDifference) = (F.n,F.n)

@testset "Misc" begin
    @testset "Kron" begin
        n = 10
        h = 1/n
        D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2

        @time D_xx = BandedBlockBandedMatrix(Kron(D², Eye(n)))
        @time D_yy = BandedBlockBandedMatrix(Kron(Eye(n),D²))
        @time Δ = D_xx + D_yy

        @test Δ isa BandedBlockBandedMatrix
        @test blockbandwidths(Δ) == subblockbandwidths(Δ) == (1,1)
        @test Δ == kron(Matrix(D²), Matrix(I,n,n)) + kron(Matrix(I,n,n), Matrix(D²))

        n = 10
        D² = FiniteDifference(n)
        D̃_xx = Kron(D², Eye(n))
        @test blockbandwidths(D̃_xx) == (1,1)
        @test subblockbandwidths(D̃_xx) == (0,0)

        V = view(D̃_xx, Block(1,1))
        @test bandwidths(V) == (0,0)

        @test BandedBlockBandedMatrix(D̃_xx) ≈ D_xx

        D̃_yy = Kron(Eye(n), D²)
        @test blockbandwidths(D̃_yy) == (0,0)
        @test subblockbandwidths(D̃_yy) == (1,1)

        V = view(D̃_yy, Block(1,1))
        @test bandwidths(V) == (1,1)

        @test BandedBlockBandedMatrix(D̃_yy) ≈ D_yy
    end

    @testset "Diagonal interface" begin
        n = 10
        h = 1/n
        D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2
        D_xx = BandedBlockBandedMatrix(Kron(D², Eye(n)))

        D = Diagonal(randn(n^2))
        @test blocksizes(D) == BlockSizes([n^2], [n^2])
        @test blockbandwidths(D) == subblockbandwidths(D) == (0,0)

        PD = PseudoBlockArray(D, blocksizes(D_xx).block_sizes)
        @test blockbandwidths(PD) == bandwidths(PD) == (0,0)

        V = view(PD, Block(1,1))
        @test bandwidths(V) == (0,0)
        @test Broadcast.BroadcastStyle(typeof(V)) == BandedStyle()

        @test D_xx + D isa BandedBlockBandedMatrix
        @test blockbandwidths(D_xx + D) == blockbandwidths(D_xx)
        @test subblockbandwidths(D_xx + D) == subblockbandwidths(D_xx)
        @test D_xx + D == Matrix(D_xx) + D

        @test D_xx - D isa BandedBlockBandedMatrix
        @test blockbandwidths(D_xx - D) == blockbandwidths(D_xx)
        @test subblockbandwidths(D_xx - D) == subblockbandwidths(D_xx)
        @test D_xx - D == Matrix(D_xx) - D

        @test D_xx*D == Matrix(D_xx)*D
        @test D_xx*D isa BandedBlockBandedMatrix
        @test blockbandwidths(D_xx*D) == blockbandwidths(D_xx)
        @test subblockbandwidths(D_xx*D) == subblockbandwidths(D_xx)

        @test D*D_xx == D*Matrix(D_xx)
        @test D*D_xx isa BandedBlockBandedMatrix
        @test blockbandwidths(D*D_xx) == blockbandwidths(D_xx)
        @test subblockbandwidths(D*D_xx) == subblockbandwidths(D_xx)
    end
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


