using BandedMatrices, BlockBandedMatrices, LinearAlgebra, LazyArrays, Test
    import LazyArrays: MemoryLayout

@testset "general" begin
    N = 10
    A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (1,1))
        A.data .= randn.()
        n = size(A,1)
    B = Matrix{Float64}(undef, n,n)
    B .= exp.(A)
    @test B == exp.(Matrix(A)) == exp.(A)
    @test exp.(A) isa Matrix
    @test A .+ 1 isa Matrix

    A = BandedBlockBandedMatrix{Float64}(undef, (1:N,1:N), (1,1), (1,1))
        A.data .= randn.()
        n = size(A,1)
    B = Matrix{Float64}(undef, n,n)
    B .= exp.(A)
    @test B == exp.(Matrix(A)) == exp.(A)
    @test exp.(A) isa Matrix
    @test A .+ 1 isa Matrix
end

@testset "lmul!/rmul!" begin
    N = 10
    A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (1,1))
        A.data .= randn.()
    B = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,2))
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

    A = BandedBlockBandedMatrix{Float64}(undef, (1:N,1:N), (1,1),(1,1))
        A.data .= randn.()
    B = BandedBlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,2),(2,2))
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
    A = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (1,1))
        A.data .= randn.()
    B = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,2))
        B.data .= randn.()
    C = BlockBandedMatrix{Float64}(undef, (1:N,1:N), (3,3))
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
    @test blockbandwidths(2A+B) == blockbandwidths(2.0.*A .+ B) == (2,2)
    B .= 2.0 .* A .+ B
    @test B == C

    N = 10
    A = BandedBlockBandedMatrix{Float64}(undef, (1:N,1:N), (1,1), (1,1))
        A.data .= randn.()
    B = BandedBlockBandedMatrix{Float64}(undef, (1:N,1:N), (2,2), (2,2))
        B.data .= randn.()
    C = BandedBlockBandedMatrix{Float64}(undef, (1:N,1:N), (3,3), (3,3))
    @time C .= A .+ B
    @test C == A + B == A .+ B

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
