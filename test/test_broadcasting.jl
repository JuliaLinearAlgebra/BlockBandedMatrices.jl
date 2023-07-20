using ArrayLayouts
using BandedMatrices
using BlockArrays
using BlockBandedMatrices
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
end
