using BlockBandedMatrices, LinearAlgebra

@testset "BlockBandedMatrix QR" begin
    N = 10
    A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,1))
    A.data .= randn.()

    F = qr(A)
    @test F.factors ≈ qr(Matrix(A)).factors
    @test F.τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ
end