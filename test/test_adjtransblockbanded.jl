using BlockBandedMatrices

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
