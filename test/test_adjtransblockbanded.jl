module TestAdjTransBlockBanded

using ArrayLayouts, BlockBandedMatrices, Test, FillArrays, BlockArrays
import BlockBandedMatrices: BandedBlockBandedRowMajor, BandedBlockBandedRows,
                            BandedBlockBandedColumns, BlockBandedRows,
                            BlockBandedColumns, blockcolsupport, blockrowsupport

@testset "Adj/Trans" begin
    @testset "BandedBlockBanded" begin
        A = BandedBlockBandedMatrix(randn(ComplexF64,10,14), 1:4,2:5, (1,2), (2,1))

        @test MemoryLayout(transpose(A)) isa BandedBlockBandedRowMajor
        @test MemoryLayout(A') isa BandedBlockBandedRows
        @test MemoryLayout(transpose(A)') isa BandedBlockBandedColumns

        @test A'[Block(1,1)] == A[Block(1,1)]'
        @test A'[Block(2,3)] == A[Block(3,2)]'
        @test transpose(A)[Block(1,1)] == transpose(A[Block(1,1)])
        @test transpose(A)[Block(2,3)] == transpose(A[Block(3,2)])

        @test blockbandwidths(A') == blockbandwidths(transpose(A)) == (2,1)
        @test subblockbandwidths(A') == subblockbandwidths(transpose(A)) == (1,2)

        @test BandedBlockBandedMatrix(A') == A'
        @test BandedBlockBandedMatrix(transpose(A)) == transpose(A)
    end

    @testset "BlockBanded" begin
        A = BlockBandedMatrix(randn(ComplexF64,10,14), 1:4,2:5, (1,2))

        @test MemoryLayout(transpose(A)) isa BlockBandedRows
        @test MemoryLayout(A') isa BlockBandedRows
        @test MemoryLayout(transpose(A)') isa BlockBandedColumns

        @test A'[Block(1,1)] == A[Block(1,1)]'
        @test A'[Block(2,3)] == A[Block(3,2)]'
        @test transpose(A)[Block(1,1)] == transpose(A[Block(1,1)])
        @test transpose(A)[Block(2,3)] == transpose(A[Block(3,2)])

        @test blockbandwidths(A') == blockbandwidths(transpose(A)) == (2,1)

        @test BlockBandedMatrix(A') == A'
        @test BlockBandedMatrix(transpose(A)) == transpose(A)
    end

    @testset "blockcolsupport" begin
        D_y = BandedBlockBandedMatrix{Float64}(undef, Fill(2,81), [3; Fill(2,79)], (0,0), (0,1))
        @test colsupport(D_y, 161) == 159:160
        @test rowsupport(D_y, 162) == 162:161
        @test colsupport(D_y', 162) == 162:161
        @test rowsupport(D_y', 161) == 159:160

        @test blockcolsupport(D_y, Block(80)) == Block.(80:80)
        @test blockrowsupport(D_y, Block(81)) == Block.(81:80)
        @test blockcolsupport(D_y', Block(81)) == Block.(81:80)
        @test blockrowsupport(D_y', Block(80)) == Block.(80:80)

        E1 = BandedBlockBandedMatrix{Float64}(undef, Int[], [1], (0,0), (0,1))
        @test colsupport(E1, 1) == 1:0
        @test rowsupport(E1', 1) == 1:0

        E2 = BandedBlockBandedMatrix{Float64}(undef, [1], Int[], (0,0), (0,1))
        @test rowsupport(E2, 1) == 1:0
        @test colsupport(E2', 1) == 1:0
    end
end

end # module
