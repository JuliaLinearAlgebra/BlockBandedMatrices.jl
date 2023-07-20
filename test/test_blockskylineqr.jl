using BlockBandedMatrices
using BlockArrays
using LinearAlgebra
using MatrixFactorizations
using Test

import BlockBandedMatrices: blockcolsupport

@testset "BlockBandedMatrix QR/QL" begin
    @testset "Square QR" begin
        N = 5
        A = BlockBandedMatrix{Float64}(undef, 1:N,1:N, (2,1))
        A.data .= randn.()

        F = qr(A)
        @test F isa MatrixFactorizations.QR{Float64,<:BlockSkylineMatrix}
        @test F.factors ≈ qr(Matrix(A)).factors
        @test F.τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ

        Q,R = F
        Q̃,R̃ = qr(Matrix(A))
        @test R ≈ R̃
        @test Matrix(Q) ≈ Matrix(Q̃)

        b = randn(size(A,1))
        @test Q'b ≈ Q̃'b
        @test Q*b ≈ Q̃*b

        @test factorize(A) isa MatrixFactorizations.QR{Float64,<:BlockSkylineMatrix}
        @test F\b ≈ ldiv!(F, copy(b)) ≈ Matrix(A)\b ≈ A\b
    end

    @testset "Thin QR" begin
        N = 5
        A = BlockBandedMatrix{Float64}(undef, 1:N+1,1:N, (2,1))
        A.data .= randn.()

        F = qr(A)
        @test F.factors ≈ qr(Matrix(A)).factors
        @test F.τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ

        Q,R = F
        Q̃,R̃ = qr(Matrix(A))
        @test R ≈ R̃
        @test Matrix(Q) ≈ Matrix(Q̃)

        b = randn(size(A,1))
        @test Q'b ≈ Q̃'b
        @test Q*b ≈ Q̃*b

        @test Q'b ≈ lmul!(Q', copy(b)) ≈ Q̃'*b

        @test F\b ≈ ldiv!(F, copy(b))[1:15] ≈ Matrix(A)\b ≈ A\b
    end

    @testset "Wide QR" begin
        N = 5
        A = BlockBandedMatrix{Float64}(undef, 1:N,1:N+1, (2,1))
        A.data .= randn.()

        F = qr(A)
        @test F.factors ≈ qr(Matrix(A)).factors
        @test F.τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ

        Q,R = F
        Q̃,R̃ = qr(Matrix(A))
        @test R ≈ R̃
        @test Matrix(Q) ≈ Matrix(Q̃)

        b = randn(size(A,1))
        @test Q'b ≈ Q̃'b
        @test Q*b ≈ Q̃*b
        Q'b
        # @test_broken DimensionMismatch ldiv!(F, copy(b))

        @test_broken F\b ≈ ldiv!(F, copy(b)) ≈ Matrix(A)\b ≈ A\b
    end

    @testset "Square QL" begin
        N = 5
        A = BlockBandedMatrix{Float64}(undef, 1:N,1:N, (2,1))
        A.data .= randn.()

        F = ql(A)
        @test F.factors ≈ ql(Matrix(A)).factors
        @test F.τ ≈ MatrixFactorizations.qlfactUnblocked!(Matrix(A)).τ

        Q,L = F
        Q̃,L̃ = ql(Matrix(A))
        @test L ≈ L̃
        @test Matrix(Q) ≈ Matrix(Q̃)

        b = randn(size(A,1))
        @test Q'b ≈ Q̃'b
        @test Q*b ≈ Q̃*b

        b = randn(size(A,1))
        @test F\b ≈ ldiv!(F, copy(b)) ≈ Matrix(A)\b ≈ A\b
    end

    @testset "Thin QL" begin
        N = 5
        A = BlockBandedMatrix{Float64}(undef, 1:N+1,1:N, (2,1))
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
        A = BlockBandedMatrix{Float64}(undef, 1:N,1:N+1, (2,1))
        A.data .= randn.()

        @test_throws ArgumentError ql(A)
    end

    @testset "Complex QR/QL" begin
        @testset "Simple" begin
            A=BlockBandedMatrix{Float64}(I, [1,1],[1,1], (0,0))
            Af=qr(A)
            B=BlockBandedMatrix{ComplexF64}(I, [1,1],[1,1], (0,0))
            Bf=qr(B)
            @test Af.factors == Bf.factors
            @test Af.τ == Bf.τ
        end

        @testset "Off-diagonals" begin
            for qrl in [qr,ql]
                for T in [Float64, ComplexF64]
                    A = BlockBandedMatrix{T}(undef, [3,2,1,3],[3,2,1,3], (1,1))
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

    @testset "Fast QR \\" begin
        A = [1. 2; 3 4]; A = A + A'
        B = [5. 6; 7 8]
        N = 10_000;
        T = BlockBandedMatrix(mortar(Tridiagonal(fill(Matrix(B'),N-1), fill(zeros(2,2),N), fill(B,N-1))))
        z = 1+2im
        @time F = qr(T - z*I);
        @test blockbandwidths(UpperTriangular(F.factors)) == (0,2)
        @test blockcolsupport(UpperTriangular(F.factors),Block(4)) == Block.(2:4)
        V = view(F.factors,Block.(1:N), Block.(1:N))
        @test blocksize(V) == (N,N)
        @test @allocated(blocksize(V)) ≤ 40
        @test blockbandwidths(V) == (1,2)
        @test @allocated(blockbandwidths(V)) ≤ 40
        R = UpperTriangular(V)
        @test axes(R) == axes(V) == axes(F.factors)
        @test blocksize(R) == (N,N)
        @test @allocated(blocksize(R)) ≤ 40
        @test blockbandwidths(R) == (0,2)
        @test @allocated(blockbandwidths(R)) ≤ 40
        @test blockcolsupport(R,Block(4)) == Block.(2:4)
        @test @allocated(blockcolsupport(R,Block(4))) ≤ 40

        b = [1; zeros(size(T,1)-1)]
        B = [Matrix(I,2,2); zeros(size(T,1)-2,2)]
        @test ((T - z*I)\b)[1] ≈ (F\b)[1] ≈ (F \ B)[1,1] ≈ ((T - z*I)\B)[1,1] ≈ -0.1309123477325813 + 0.28471699370329884im
    end

    @testset "BigFloat QR" begin
        N = 5
        A = BlockBandedMatrix{BigFloat}(undef, 1:N,1:N, (2,1))
        A.data .= randn.()
        @test Matrix(qr(A).Q) ≈ Matrix(qr(Float64.(A)).Q)
        b = randn(size(A,1))
        @test qr(A .+ 0im)\b ≈ qr(A)\b ≈ A\b
    end
end



# N = 500;
# A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,1));
# A.data .= randn.();
# @time F = qr(A); # 11s
# b = randn(size(A,1));
# @time F\b; # 0.6s
