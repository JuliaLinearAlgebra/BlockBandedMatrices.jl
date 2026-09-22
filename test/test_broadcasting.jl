module TestBroadcasting

using ArrayLayouts
using BandedMatrices
using BlockArrays
using BlockBandedMatrices
using FillArrays
using LinearAlgebra
using Test

import Base: oneto

@testset "broadcasting" begin
    @testset "general" begin
        N = 10
        A = BlockBandedMatrix{Float64}(undef, 1:N,1:N, (1,1))
        A.data .= randn.()
        n = size(A,1)
        B = Matrix{Float64}(undef, n,n)
        B .= exp.(A)
        @test B == exp.(Matrix(A)) == exp.(A)
        @test exp.(A) isa BlockBandedMatrix
        @test A .+ 1 isa BlockBandedMatrix

        A = BandedBlockBandedMatrix{Float64}(undef, 1:N,1:N, (1,1), (1,1))
            A.data .= randn.()
            n = size(A,1)
        B = Matrix{Float64}(undef, n,n)
        B .= exp.(A)
        @test B == exp.(Matrix(A)) == exp.(A)
        @test exp.(A) isa BandedBlockBandedMatrix
        @test A .+ 1 isa BandedBlockBandedMatrix
    end

    @testset "lmul!/rmul!" begin
        N = 10
        A = BlockBandedMatrix{Float64}(undef, oneto(N),oneto(N), (1,1))
            A.data .= randn.()
        B = BlockBandedMatrix{Float64}(undef, oneto(N),oneto(N), (2,2))
        B .= (-).(A)
        @test similar(A) isa typeof(A)
        @test similar(A,Float64) isa typeof(A)
        @test -A isa typeof(A)
        @test (-).(A) isa typeof(A)
        @test blockbandwidths(A) == blockbandwidths(-A) == blockbandwidths((-).(A))
        @test B == -A == (-).(A)
        @test A-I isa typeof(A)
        @test I-A isa typeof(A)
        @test blockbandwidths(A) == blockbandwidths(A-I) == blockbandwidths(I-A)

        B .= 2.0.*A

        @test B ==  2A == 2.0.*A
        @test 2A isa typeof(A)
        @test 2.0.*A isa typeof(A)
        @test blockbandwidths(2A) == blockbandwidths(2.0.*A) == blockbandwidths(A)

        A .= 2.0.*A
        @test A == B

        B .= A.*2.0

        @test B ==  A*2 == A.*2.0
        @test A*2 isa typeof(A)
        @test A .* 2.0 isa typeof(A)
        @test blockbandwidths(A*2) == blockbandwidths(A.*2.0) == blockbandwidths(A)
        A .= A.*2.0
        @test A == B

        B .= A ./ 2.0

        @test B == A/2 == A ./ 2.0
        @test A/2 isa typeof(A)
        @test A ./ 2.0 isa typeof(A)
        @test blockbandwidths(A/2) == blockbandwidths(A ./ 2.0) == blockbandwidths(A)

        B .= 2.0 .\ A

        @test B == A/2 == A ./ 2.0
        @test 2\A isa typeof(A)
        @test 2.0 .\ A isa typeof(A)
        @test blockbandwidths(2\A) == blockbandwidths(2.0 .\ A) == blockbandwidths(A)

        A = BandedBlockBandedMatrix{Float64}(undef, oneto(N),oneto(N), (1,1),(1,1))
            A.data .= randn.()
        B = BandedBlockBandedMatrix{Float64}(undef, oneto(N),oneto(N), (2,2),(2,2))
        B .= (-).(A)
        @test similar(A) isa typeof(A)
        @test similar(A,Float64) isa typeof(A)
        @test -A isa typeof(A)
        @test (-).(A) isa typeof(A)
        @test blockbandwidths(A) == blockbandwidths(-A) == blockbandwidths((-).(A))
        @test B == -A == (-).(A)
        @test A-I isa typeof(A)
        @test I-A isa typeof(A)
        @test bandwidths(A) == bandwidths(A-I) == bandwidths(I-A)

        B .= 2.0.*A

        @test B ==  2A == 2.0.*A
        @test 2A isa typeof(A)
        @test 2.0.*A isa typeof(A)
        @test blockbandwidths(2A) == blockbandwidths(2.0.*A) == blockbandwidths(A)
        @test subblockbandwidths(2A) == subblockbandwidths(2.0.*A) == subblockbandwidths(A)

        A .= 2.0.*A
        @test A == B

        B .= A.*2.0

        @test B ==  A*2 == A.*2.0
        @test A*2 isa typeof(A)
        @test A .* 2.0 isa typeof(A)
        @test blockbandwidths(A*2) == blockbandwidths(A.*2.0) == blockbandwidths(A)
        @test subblockbandwidths(A*2) == subblockbandwidths(A.*2.0) == subblockbandwidths(A)
        A .= A.*2.0
        @test A == B

        B .= A ./ 2.0

        @test B == A/2 == A ./ 2.0
        @test A/2 isa typeof(A)
        @test A ./ 2.0 isa typeof(A)
        @test blockbandwidths(A/2) == blockbandwidths(A ./ 2.0) == blockbandwidths(A)
        @test subblockbandwidths(A/2) == subblockbandwidths(A ./ 2.0) == subblockbandwidths(A)

        B .= 2.0 .\ A

        @test B == A/2 == A ./ 2.0
        @test 2\A isa typeof(A)
        @test 2.0 .\ A isa typeof(A)
        @test blockbandwidths(2\A) == blockbandwidths(2.0 .\ A) == blockbandwidths(A)
        @test subblockbandwidths(2\A) == subblockbandwidths(2.0 .\ A) == subblockbandwidths(A)
    end

    @testset "axpy!" begin
        N = 10
        A = BlockBandedMatrix{Float64}(undef, 1:N,1:N, (1,1))
        A.data .= randn.()
        B = BlockBandedMatrix{Float64}(undef, 1:N,1:N, (2,2))
        B.data .= randn.()
        C = BlockBandedMatrix{Float64}(undef, 1:N,1:N, (3,3))
        @time C .= A .+ B
        @test C == A + B == A .+ B

        @test A + B isa typeof(A)
        @test A .+ B isa typeof(A)
        @test blockbandwidths(A+B) == blockbandwidths(A.+B) == (2,2)
        @time B .= A .+ B
        @test B == C

        C .= 2.0 .* A .+ B
        @test C == 2A+B == 2.0.*A .+ B
        @test 2A + B isa typeof(A)
        @test 2.0.*A .+ B isa typeof(A)
        bc = Base.broadcasted(+, Base.broadcasted(*, 2.0, A), B)
        blockbandwidths(bc)
        @test blockbandwidths(2A+B) == blockbandwidths(2.0.*A .+ B) == (2,2)
        B .= 2.0 .* A .+ B
        @test B == C

        N = 10
        A = BandedBlockBandedMatrix{Float64}(undef, Base.OneTo(N),Base.OneTo(N), (1,1), (1,1))
            A.data .= randn.()
        B = BandedBlockBandedMatrix{Float64}(undef, Base.OneTo(N),Base.OneTo(N), (2,2), (2,2))
            B.data .= randn.()
        C = BandedBlockBandedMatrix{Float64}(undef, Base.OneTo(N),Base.OneTo(N), (3,3), (3,3))
        @time C .= A .+ B
        @test C == A + B == A .+ B

        bc = Base.broadcasted(+, A, B)
        @test @inferred(axes(bc)) === axes(A)

        @test A + B isa typeof(A)
        @test A .+ B isa typeof(A)
        @test blockbandwidths(A+B) == blockbandwidths(A.+B) == (2,2)
        @test subblockbandwidths(A+B) == subblockbandwidths(A.+B) == (2,2)
        @time B .= A .+ B
        @test B == C


        C .= 2.0 .* A .+ B
        @test C == 2A+B == 2.0.*A .+ B
        @test 2A + B isa typeof(A)
        @test 2.0.*A .+ B isa typeof(A)
        @test blockbandwidths(2A+B) == blockbandwidths(2.0.*A .+ B) == (2,2)
        @test subblockbandwidths(2A+B) == subblockbandwidths(2.0.*A .+ B) == (2,2)
        B .= 2.0 .* A .+ B
        @test B == C
    end

    @testset "Degenerate bands" begin
        A = BandedBlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (2,0), (1,1)); A.data .= randn.();
        B = BandedBlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (1,-1), (1,1)); B.data .= randn.();
        @test A + B == B + A == Matrix(A) + Matrix(B)
        @test A - B == Matrix(A) - Matrix(B)
        @test B - A == Matrix(B) - Matrix(A)

        A = BandedBlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (0,2), (1,1)); A.data .= randn.();
        B = BandedBlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (-1,1), (1,1)); B.data .= randn.();
        @test A + B == B + A == Matrix(A) + Matrix(B)
        @test A - B == Matrix(A) - Matrix(B)
        @test B - A == Matrix(B) - Matrix(A)

        A = BlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (2,0)); A.data .= randn.();
        B = BandedBlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (1,-1), (1,1)); B.data .= randn.();
        @test A + B == B + A == Matrix(A) + Matrix(B)
        @test A - B == Matrix(A) - Matrix(B)
        @test B - A == Matrix(B) - Matrix(A)

        A = BlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (2,0)); A.data .= randn.();
        B = BlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (1,-1)); B.data .= randn.();
        @test A + B == B + A == Matrix(A) + Matrix(B)
        @test A - B == Matrix(A) - Matrix(B)
        @test B - A == Matrix(B) - Matrix(A)
    end

    @testset "Diag" begin
        @testset "BlockBanded" begin
            A = BlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (2,1)); A.data .= randn.();
            b = 1:size(A,1)
            bc = Base.broadcasted(*, b, A)
            @test blockisequal(axes(bc), axes(A))
            @test blockaxes(bc) == blockaxes(A)
            @test blocksize(bc) == blocksize(A)
            @test blockbandwidths(bc) == (2,1)
            @test b .* A == b .* Matrix(A)
            @test b .* A isa BlockBandedMatrix
            @test blockisequal(axes(b .* A), axes(A))

            bᵗ = permutedims(1:size(A,2))
            bc = Base.broadcasted(*, A, bᵗ)
            @test blockisequal(axes(bc), axes(A))
            @test blockaxes(bc) == blockaxes(A)
            @test blocksize(bc) == blocksize(A)
            @test blockbandwidths(bc) == (2,1)
            @test A .* bᵗ == Matrix(A) .* bᵗ
            @test A .* bᵗ isa BlockBandedMatrix
            @test blockisequal(axes(A .* bᵗ), axes(A))
        end
        @testset "BandedBlockBanded" begin
            A = BandedBlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (2,1), (1,2)); A.data .= randn.();
            b = 1:size(A,1)
            bc = Base.broadcasted(*, b, A)
            @test blockisequal(axes(bc), axes(A))
            @test blockaxes(bc) == blockaxes(A)
            @test blocksize(bc) == blocksize(A)
            @test b .* A == b .* Matrix(A)
            @test b .* A isa BandedBlockBandedMatrix
            @test blockisequal(axes(b .* A), axes(A))
            @test blockbandwidths(bc) == blockbandwidths(b .* A) == (2,1)
            @test subblockbandwidths(bc) == subblockbandwidths(b .* A) == (1,2)

            bᵗ = permutedims(1:size(A,2))
            bc = Base.broadcasted(*, A, bᵗ)
            @test blockisequal(axes(bc), axes(A))
            @test blockaxes(bc) == blockaxes(A)
            @test blocksize(bc) == blocksize(A)
            @test blockbandwidths(bc) == (2,1)
            @test subblockbandwidths(bc) == (1,2)
            @test A .* bᵗ == Matrix(A) .* bᵗ
            @test A .* bᵗ isa BandedBlockBandedMatrix
            @test blockisequal(axes(A .* bᵗ), axes(A))

            bc = Base.broadcasted(\, b, A)
            @test blockisequal(axes(bc), axes(A))
            @test blockaxes(bc) == blockaxes(A)
            @test blocksize(bc) == blocksize(A)
            @test b .\ A == b .\ Matrix(A)
            @test b .\ A isa BandedBlockBandedMatrix
            @test blockisequal(axes(b .* A), axes(A))
            @test blockbandwidths(bc) == blockbandwidths(b .* A) == (2,1)
            @test subblockbandwidths(bc) == subblockbandwidths(b .* A) == (1,2)

            bc = Base.broadcasted(/, A, b)
            @test blockisequal(axes(bc), axes(A))
            @test blockaxes(bc) == blockaxes(A)
            @test blocksize(bc) == blocksize(A)
            @test b ./ A == b ./ Matrix(A)
            @test b ./ A isa BandedBlockBandedMatrix
            @test blockisequal(axes(b .* A), axes(A))
            @test blockbandwidths(bc) == blockbandwidths(b .* A) == (2,1)
            @test subblockbandwidths(bc) == subblockbandwidths(b .* A) == (1,2)
        end
        @testset "Incompatible blocksize" begin
            A = BandedBlockBandedMatrix{Float64}(undef, Fill(4,4), Fill(4,3), (2,1), (1,2)); A.data .= randn.();
            C = Array{Float64}(undef, size(A))
            C .= A .+ A
            @test C == A + A
        end
    end

    @testset "blockbandwidths" begin
        B = BlockArray(ones(6,6), 1:3, 1:3)
        BB = BlockBandedMatrix(B, (1,1))
        bc = Broadcast.broadcasted(+, BB, BB)
        bbw = @inferred blockbandwidths(bc)
        @test bbw == blockbandwidths(BB)
    end

    @testset "adjoint/transpose" begin
        A = BandedBlockBandedMatrix(randn(ComplexF64,10,10), 1:4,1:4, (1,1), (1,1))
        B = BlockBandedMatrix(randn(ComplexF64,10,10), 1:4,1:4, (1,1))
        α = 2+im

        @testset "$op" for op in (adjoint, transpose)
            Wrap = op === adjoint ? Adjoint : Transpose
            @testset "$(typeof(M).name.name)" for M in (A, B)
                W, Wm = op(M), op(Matrix(M))
                for (R,Rm) in ((α .* W, α .* Wm), (W .* α, Wm .* α), (W ./ α, Wm ./ α),
                               (α .\ W, α .\ Wm), ((-).(W), (-).(Wm)),
                               (W .+ W, Wm .+ Wm), (W .- W, Wm .- Wm),
                               (α .* W .+ W, α .* Wm .+ Wm))
                    @test R isa Wrap{ComplexF64}
                    @test parent(R) isa typeof(M).name.wrapper
                    @test R ≈ Rm
                end
            end
            @test op(A) .+ op(B) isa Wrap{ComplexF64}
            @test parent(op(A) .+ op(B)) isa BlockBandedMatrix
            @test op(A) .+ op(B) ≈ op(Matrix(A)) .+ op(Matrix(B))
        end

        # the wrapper is dropped when it cannot be moved through the broadcast
        @test exp.(A') ≈ exp.(Matrix(A)')
        @test !(exp.(A') isa Adjoint)
        @test imag.(A') ≈ imag.(Matrix(A)')
        @test !(imag.(A') isa Adjoint)
        @test A' .+ transpose(A) ≈ Matrix(A)' .+ transpose(Matrix(A))
        @test !(A' .+ transpose(A) isa Adjoint)
        @test A' .+ A ≈ Matrix(A)' .+ Matrix(A)
        @test A' .* ones(10,10) ≈ Matrix(A)'

        f(α, A) = α .* A'
        @test @inferred(f(α, A)) isa Adjoint{ComplexF64,<:BandedBlockBandedMatrix}
        @test f(α, A) ≈ α .* Matrix(A)'

        C = BandedBlockBandedMatrix{ComplexF64}(undef, 1:4,1:4, (1,1), (1,1))
        C .= α .* A'
        @test C ≈ α .* Matrix(A)'
        D = Matrix{ComplexF64}(undef, 10, 10)
        D .= α .* A'
        @test D ≈ α .* Matrix(A)'
    end

    @testset "data" begin
        A = BandedBlockBandedMatrix{Float64}(undef, 1:4,1:4, (1,1),(1,1))
        B = BandedBlockBandedMatrix{Float64}(undef, 1:4,1:4, (1,1),(1,1))
        # the data outside the bands is junk that must not leak into the result
        A.data .= NaN; B.data .= NaN
        for M in (A,B), J = 1:4, K = max(1,J-1):min(4,J+1)
            V = view(M, Block(K), Block(J))
            for j = axes(V,2), k = colrange(V,j)
                V[k,j] = randn()
            end
        end
        Am, Bm = Matrix(A), Matrix(B)
        α = 2.0
        @test !any(isnan, Am)

        for (R,Rm) in ((α .* A, α .* Am), (A .* α, Am .* α), (A ./ α, Am ./ α), (α .\ A, α .\ Am),
                       ((-).(A), (-).(Am)), (A .+ B, Am .+ Bm), (A .- B, Am .- Bm),
                       (A .* B, Am .* Bm), (α .* A .+ B, α .* Am .+ Bm))
            @test R isa BandedBlockBandedMatrix
            @test blockbandwidths(R) == (1,1)
            @test subblockbandwidths(R) == (1,1)
            @test Matrix(R) == Rm
        end
        @test typeof(α .* A) == typeof(similar(A, Float64))

        # the data is only used when the structures line up
        C = BandedBlockBandedMatrix(randn(10,10), 1:4,1:4, (2,1),(1,0)); Cm = Matrix(C)
        @test A .+ C ≈ Am .+ Cm
        @test blockbandwidths(A .+ C) == (2,1)
        @test subblockbandwidths(A .+ C) == (1,1)
        @test A .* C ≈ Am .* Cm
        @test blockbandwidths(A .* C) == (1,1)
        @test subblockbandwidths(A .* C) == (1,0)
        @test_throws DimensionMismatch A .+ BandedBlockBandedMatrix(randn(10,14), 1:4,2:5, (1,1),(1,1))

        # broadcasts that do not preserve the bands are unchanged
        @test A .+ 1 ≈ Am .+ 1
        @test exp.(A) ≈ exp.(Am)
        @test Diagonal(1:10) .* A ≈ Diagonal(1:10) .* Am

        f(α, A) = α .* A
        g(A, B) = A .+ B
        h(α, A, B) = α .* A .+ B
        @test @inferred(f(α, A)) ≈ α .* Am
        # Following can't be inferred because of issues in type-inferrence
        # see Base.Broadcast.axistype overload in
        #  BlockArrays/src/blockbroadcast.jl:45
        @test g(A, B) ≈ Am .+ Bm
        @test h(α, A, B) ≈ α .* Am .+ Bm

        # and the data is used through an adjoint as well
        @test parent((2+im) .* A') isa BandedBlockBandedMatrix
        @test (2+im) .* A' ≈ (2+im) .* Am'
    end
end

end # module
