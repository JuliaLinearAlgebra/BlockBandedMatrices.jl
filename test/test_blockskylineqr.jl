using BlockBandedMatrices, LinearAlgebra, MatrixFactorizations

@testset "BlockBandedMatrix QR" begin
    N = 5
    A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,1))
    A.data .= randn.()

    F = qr(A)
    @test F.factors ≈ qr(Matrix(A)).factors
    @test F.τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ

    Q,R = F
    Q̃,R̃ = qr(Matrix(A))
    @test R ≈ R̃
    @test Q ≈ Q̃
    
    b = randn(size(A,1))
    @test Q'b ≈ Q̃'b
    @test Q*b ≈ Q̃*b
    
    @test F\b ≈ ldiv!(F, copy(b)) ≈ Matrix(A)\b ≈ A\b
end

@testset "BlockBandedMatrix QL" begin
    N = 5
    A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,1))
    A.data .= randn.()

    F = ql(A)
    @test F.factors ≈ ql(Matrix(A)).factors
    @test F.τ ≈ MatrixFactorizations.qlfactUnblocked!(Matrix(A)).τ

    Q,L = F
    Q̃,L̃ = ql(Matrix(A))
    @test L ≈ L̃
    @test Q ≈ Q̃
    
    b = randn(size(A,1))
    @test Q'b ≈ Q̃'b
    @test Q*b ≈ Q̃*b

    b = randn(size(A,1))
    @test F\b ≈ ldiv!(F, copy(b)) ≈ Matrix(A)\b ≈ A\b
end


# N = 500;
# A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,1));
# A.data .= randn.();
# @time F = qr(A); # 11s
# b = randn(size(A,1));
# @time F\b; # 0.6s

