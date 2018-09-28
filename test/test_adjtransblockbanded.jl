using BlockBandedMatrices
using BlockBandedMatrices: BlockBandedSizes, BandedBlockBandedSizes
using BlockArrays: BlockSizes

@testset "Adj/Trans" begin
    A = BandedBlockBandedMatrix(randn(ComplexF64,10,14), (1:4,2:5), (1,2), (2,1))

    @test A'[Block(1,1)] == A[Block(1,1)]'
    @test A'[Block(2,3)] == A[Block(3,2)]'
    @test transpose(A)[Block(1,1)] == transpose(A[Block(1,1)])
    @test transpose(A)[Block(2,3)] == transpose(A[Block(3,2)])

    @test blockbandwidths(A') == blockbandwidths(transpose(A)) == (2,1)
    @test subblockbandwidths(A') == subblockbandwidths(transpose(A)) == (1,2)

    @test BandedBlockBandedMatrix(A') == A'
    @test BandedBlockBandedMatrix(transpose(A)) == transpose(A)
end

@testset "blocksize transpose" begin
    a = BlockBandedSizes(BlockSizes(rand(1:10, 3), rand(1:10, 4)), rand(1:10), rand(1:10))
    @test transpose(a).block_sizes.cumul_sizes[1] == a.block_sizes.cumul_sizes[2]
    @test transpose(a).block_sizes.cumul_sizes[2] == a.block_sizes.cumul_sizes[1]
    @test blockbandwidth(transpose(a), 2) == blockbandwidth(a, 1)
    @test blockbandwidth(transpose(a), 1) == blockbandwidth(a, 2)

    a = BandedBlockBandedSizes(BlockSizes(rand(1:10, 3), rand(1:10, 4)),
                               rand(1:10), rand(1:10), rand(1:10), rand(1:10))
    @test transpose(a).block_sizes.cumul_sizes[1] == a.block_sizes.cumul_sizes[2]
    @test transpose(a).block_sizes.cumul_sizes[2] == a.block_sizes.cumul_sizes[1]
    @test transpose(a).l == a.u
    @test transpose(a).u == a.l
    @test transpose(a).λ == a.μ
    @test transpose(a).μ == a.λ
end
