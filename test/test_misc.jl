using LazyArrays, BlockBandedMatrices, BandedMatrices, Test
import Base: getindex, size
import BandedMatrices: bandwidths, AbstractBandedMatrix, BandedStyle

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

# TODO: move into code
Base.BroadcastStyle(::Type{<:Eye}) = BandedStyle()


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
    K = Kron(D², Eye(n))
    @test blockbandwidths(K) == (1,1)
    @test subblockbandwidths(K) == (0,0)

    V = view(K, Block(1,1))
    @test Base.BroadcastStyle(typeof(V)) == BandedStyle()
    @test bandwidths(V) == (0,0)

    @test BandedBlockBandedMatrix(K) == D_xx
end
