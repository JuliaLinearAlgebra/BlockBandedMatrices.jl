using LazyArrays, BlockBandedMatrices, Test

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
end
