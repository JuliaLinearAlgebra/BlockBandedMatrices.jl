using BlockBandedMatrices, LinearAlgebra, MatrixFactorizations

@testset "BlockBandedMatrix QR/QL" begin
    @testset "Square QR" begin
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

    @testset "Thin QR" begin
        N = 5
        A = BlockBandedMatrix{Float64}(undef, (1:N+1,1:N), (2,1))
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

        @test Q'b ≈ lmul!(Q', copy(b)) ≈ Q̃'*b

        @test F\b ≈ ldiv!(F, copy(b))[1:15] ≈ Matrix(A)\b ≈ A\b
    end

    @testset "Wide QR" begin
        N = 5
        A = BlockBandedMatrix{Float64}(undef, (1:N,1:N+1), (2,1))
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
        Q'b
        # @test_broken DimensionMismatch ldiv!(F, copy(b))

        @test_broken F\b ≈ ldiv!(F, copy(b)) ≈ Matrix(A)\b ≈ A\b
    end

    @testset "Square QL" begin
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

    @testset "Thin QL" begin
        N = 5
        A = BlockBandedMatrix{Float64}(undef, (1:N+1,1:N), (2,1))
        A.data .= randn.()

        # F = ql(A)
        # @test F.factors ≈ ql(Matrix(A)).factors
        # @test F.τ ≈ MatrixFactorizations.qlfactUnblocked!(Matrix(A)).τ

        # Q,L = F
        # Q̃,L̃ = ql(Matrix(A))
        # @test L ≈ L̃
        # @test Q ≈ Q̃

        # b = randn(size(A,1))
        # @test Q'b ≈ Q̃'b
        # @test Q*b ≈ Q̃*b

        # b = randn(size(A,1))
        # @test F\b ≈ ldiv!(F, copy(b)) ≈ Matrix(A)\b ≈ A\b
    end

    @testset "Wide QL" begin
        N = 5
        A = BlockBandedMatrix{Float64}(undef, (1:N,1:N+1), (2,1))
        A.data .= randn.()

        @test_throws ArgumentError ql(A)
    end

    @testset "Complex QR/QL" begin
        @testset "Simple" begin
            A=BlockBandedMatrix{Float64}(I, ([1,1],[1,1]), (0,0))
            Af=qr(A)
            B=BlockBandedMatrix{ComplexF64}(I, ([1,1],[1,1]), (0,0))
            Bf=qr(B)
            @test Af.factors == Bf.factors
            @test Af.τ == Bf.τ
        end

        @testset "Off-diagonals" begin
            for qrl in [qr,ql]
                for T in [Float64, ComplexF64]
                    A = BlockBandedMatrix{T}(undef, ([3,2,1,3],[3,2,1,3]), (1,1))
                    A.data .= rand(T, size(A.data))
                    Af = qrl(A)
                    @test Af.factors isa BlockBandedMatrix

                    Am = Matrix(A)
                    Amf = qrl(Am)

                    @test Af.factors ≈ Amf.factors
                end
            end
        end
    end
end



# N = 500;
# A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,1));
# A.data .= randn.();
# @time F = qr(A); # 11s
# b = randn(size(A,1));
# @time F\b; # 0.6s
