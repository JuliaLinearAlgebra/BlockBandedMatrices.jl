using BlockBandedMatrices, LinearAlgebra

@testset "BlockBandedMatrix QR" begin
    N = 5
    A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,1))
    A.data .= randn.()

    F = qr(A)
    @test F.factors ≈ qr(Matrix(A)).factors
    @test F.τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ

    b = randn(size(A,1))
    @test F\b ≈ ldiv!(F, copy(b)) ≈ Matrix(A)\b ≈ A\b
end


# N = 500;
# A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,1));
# A.data .= randn.();
# @time F = qr(A); # 11s
# b = randn(size(A,1));
# @time F\b; # 0.6s

