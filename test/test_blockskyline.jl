
using LazyArrays, BlockBandedMatrices, LinearAlgebra, Random, Test

Random.seed!(0)

@testset "BlockSkylineMatrix" begin

    @testset "@jagot lmul! bug" begin
        rows = rand(1:10, 5)
        l = rand(-2:2, 5)
        u = rand(-2:2, 5)

        m = sum(rows)

        A = BlockSkylineMatrix(Zeros(m,m), (rows,rows), (l,u))
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
end
