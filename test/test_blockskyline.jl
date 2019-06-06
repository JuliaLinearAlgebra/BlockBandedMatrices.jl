using LazyArrays, BlockBandedMatrices, LinearAlgebra, Random, Test
import BlockBandedMatrices: colblockbandwidths,
    BroadcastStyle, BlockSkylineStyle, blocksizes

Random.seed!(0)

@testset "BlockSkylineMatrix" begin

    @testset "@jagot lmul! bug" begin
        rows = rand(1:10, 5)
        l = rand(-2:2, 5)
        u = rand(-2:2, 5)

        m = sum(rows)

        A = BlockSkylineMatrix(Zeros(m,m), (rows,rows), (l,u))
        @test @inferred(size(A)) == (sum(rows),sum(rows))

        @test Array(A) == Matrix(A)

        A.data .= rand(size(A.data)...)

        V = zeros(m,2)
        V[:,1] .= rand(m)
        reference = A*V[:,1]

        @view(V[:,2]) .= Mul(A, @view(V[:,1]))
        @test V[:,2] ≈ reference

        V[:,2] .= NaN
        @view(V[:,2]) .= Mul(A, @view(V[:,1]))
        @test V[:,2] ≈ reference
    end

    @testset "BlockSkylineMatrix multiplication" begin
        rows = [3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3]
        l,u = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]

        M = BlockSkylineMatrix{Float64}(undef, (rows,rows), (l,u))
        M.data .= 1

        d = Diagonal(1.0:size(M,2))
        D = BandedBlockBandedMatrix(d, (rows,rows), (0,0), (0,0))

        MD = M*D
        @test MD isa BlockSkylineMatrix
        @test MD == Matrix(M)*d
        @test colblockbandwidths(MD) == (l,u)

        MM = M*M
        @test MM isa BlockSkylineMatrix
        @test MM == Matrix(M)^2
        # Ensure correct (minimal) bandedness of product
        MMl,MMu = colblockbandwidths(MM)
        @test MMl[1:7] == [3,4,3,4,3,4,3]
        @test all(MMl[8:10] .≥ [3,2,1])
        @test all(MMu[2:4] .≥ [1,2,3])
        @test MMu[5:11] == [3,4,3,4,3,4,3]

        N = BlockBandedMatrix{Float64}(undef, (rows,rows), (1,1))
        N.data .= 1
        NN = N*N
        # We don't want a BlockBandedMatrix^2 to become a general
        # BlockSkylineMatrix
        @test NN isa BlockBandedMatrix
        @test NN == Matrix(N)^2

        rows = [9, 4, 1, 10, 6]
        O = BlockSkylineMatrix{Int64}(undef, (rows,rows), ([-2, 2, 0, 2, -1],[-1, 2, 1, 0, -1]))
        O.data .= 1
        OO = O*O
        @test OO isa BlockSkylineMatrix
        @test OO == Matrix(O)^2
    end

    @testset "Broadcasting" begin
        rows = [3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3]
        l,u = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]

        M = BlockSkylineMatrix{Float64}(undef, (rows,rows), (l,u))
        M.data .= 1

        @test BroadcastStyle(BlockSkylineMatrix) == BlockSkylineStyle()
        @test blocksizes(M + M) == blocksizes(M)

        # A is a BlockSkylineMatrix with varying bandwidths
        rows = [3,4,5]
        A = BlockSkylineMatrix{Float64}(undef, (rows,rows), ([1,0,1],[0,1,0]))
        A.data .= 1

        # B is diagonal
        B = BandedBlockBandedMatrix{Float64}(undef, (rows, rows), (0,0), (0,0))
        B.data .= -1

        # AB should have the same structure as A
        AB = A+B
        @test Matrix(AB) == Matrix(A) + Matrix(B)
        @test blocksizes(AB) == blocksizes(A)

        # C has larger bandwidths than A
        C = BandedBlockBandedMatrix{Float64}(undef, (rows, rows), (1,1), (1,1))
        C.data .= -1

        # Thus, AC should be a BlockSkylineMatrix with C's bandwidths.
        AC = A+C
        @test Matrix(AC) == Matrix(A) + Matrix(C)
        @test blocksizes(AC) == blocksizes(C)
    end
end
