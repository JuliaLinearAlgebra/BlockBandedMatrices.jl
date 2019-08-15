using BlockArrays, BandedMatrices, BlockBandedMatrices, FillArrays, LazyArrays, LinearAlgebra, Test
import BlockBandedMatrices: MemoryLayout, ColumnMajor, BroadcastStyle, BlockBandedStyle
import LazyArrays: Mul, MulAdd
import Base.Broadcast: materialize!

@testset "BlockBandedMatrix" begin
    @test BroadcastStyle(BlockBandedMatrix) == BlockBandedStyle()

    @testset "BlockBandedMatrix constructors" begin
        l , u = 1,1
        N = M = 4
        cols = rows = 1:N

        @test Matrix(BlockBandedMatrix(Zeros(sum(rows),sum(cols)), (rows, cols), (l,u))) ==
            Array(BlockBandedMatrix(Zeros(sum(rows),sum(cols)), (rows, cols), (l,u))) == 
            zeros(Float64, 10, 10)

        @test Matrix(BlockBandedMatrix{Int}(Zeros(sum(rows),sum(cols)), (rows,cols), (l,u))) ==
            zeros(Int, 10, 10)

        @test Matrix(BlockBandedMatrix(Eye(sum(rows)), (rows,cols), (l,u))) ==
            Matrix{Float64}(I, 10, 10)

        @test Matrix(BlockBandedMatrix{Int}(Eye(sum(rows)), (rows,cols), (l,u))) ==
            Matrix{Int}(I, 10, 10)

        @test Matrix(BlockBandedMatrix(I, (rows,cols), (l,u))) ==
            Matrix{Float64}(I, 10, 10)

        @test Matrix(BlockBandedMatrix{Int}(I, (rows,cols), (l,u))) ==
            Matrix{Int}(I, 10, 10)
    end

    @testset "BlockBandedMatrix block indexing" begin
        l , u = 1,1
        N = M = 4
        cols = rows = 1:N
        A = BlockBandedMatrix{Int}(undef, (rows,cols), (l,u))
            A.data .= 1:length(A.data)

        @test A[1,1] == 1
        @test A[1,3] == 10



        @test blockbandwidth(A,1)  == 1
        @test blockbandwidths(A) == (l,u)

        # check views of blocks are indexing correctly


        @test A[Block(1), Block(1)] isa Matrix
        @test A[Block(1), Block(1)] == A[Block(1,1)] == BlockArrays.getblock(A, 1, 1) == Matrix(view(A, Block(1,1)))
        @test A[1,1] == view(A,Block(1),Block(1))[1,1] == view(A,Block(1,1))[1,1] == A[Block(1,1)][1,1]  == A[Block(1),Block(1)][1,1] == 1
        @test A[2,1] == view(A,Block(2),Block(1))[1,1] == view(A,Block(2,1))[1,1] == 2
        @test A[3,1] == view(A,Block(2),Block(1))[2,1] == 3
        @test A[4,1] == 0
        @test A[1,2] == view(A,Block(1,2))[1,1] == 4
        @test A[1,3] == view(A,Block(1,2))[1,2] == view(A,Block(1,2))[2] == 10

        @test view(A, Block(3),Block(1)) ≈ [0,0,0]
        @test_throws BandError view(A, Block(3),Block(1))[1,1] = 4
        @test_throws BoundsError view(A, Block(5,1))


        # test blocks
        V = view(A, Block(1,1))
        @test_throws BoundsError V[2,1]

        V = view(A, Block(3,4))
        @test V[3,1] == 45
        V[3,1] = -7
        @test V[3,1] == -7
        @test Matrix(V) isa Matrix{Int}
        @test Matrix{Float64}(V) isa Matrix{Float64}
        @test Matrix{Float64}(Matrix(V)) == Matrix{Float64}(V)
        @test A[4:6,7:10] ≈ Matrix(V)

        A[1,1] = -5
        @test A[1,1] == -5
        A[1,3] = -6
        @test A[1,3] == -6

        A[Block(3,4)] = Matrix(Ones{Int}(3,4))
        @test A[Block(3,4)] == Matrix(Ones{Int}(3,4))

        ## Bug in setindex!
        ret = BlockBandedMatrix(Zeros{Float64}((4,6)), ([2,2], [2,2,2]), (0,2))


        V = view(ret, Block(1), Block(2))
        V[1,1] = 2
        @test ret[1,2] == 0
    end

    @testset "BlockBandedMatrix indexing" begin
        l , u = 1,1
        N = M = 10
        cols = rows = 1:N
        A = BlockBandedMatrix{Float64}(undef, (rows,cols), (l,u))
            A.data .= 1:length(A.data)

        A[1,1] = 5
        @test A[1,1] == 5

        @test_throws BandError A[1,4] = 5
        A[1,4] = 0
        @test A[1,4] == 0

        @test A[1:10,1:10] ≈ Matrix(A)[1:10,1:10]

        l , u = 2,1
        N = M = 5
        cols = rows = 1:N

        A = BlockBandedMatrix{Int}(undef, (rows,cols), (l,u))
        A.data .= 1:length(A.data)

        @test A[1,2] == 7
        A[1,2] = -5
        @test A[1,2] == -5
    end

    @testset "BlockBandedMatrix BLAS arithmetic" begin
        l , u = 1,1
        N = M = 10
        cols = rows = fill(100,N)
        A = BlockBandedMatrix{Float64}(undef, (rows,cols), (l,u))
            A.data .= 1:length(A.data)

        V = view(A, Block(N,N))
        @test MemoryLayout(typeof(V)) == ColumnMajor()

        Y = zeros(cols[N], cols[N])
        @time BLAS.axpy!(2.0, V, Y)
        @test Y ≈ 2A[Block(N,N)]

        Y = BandedMatrix(Zeros(cols[N], cols[N]), (0, 0))
        @test_throws BandError BLAS.axpy!(2.0, V, Y)

        AN = A[Block(N,N)]
        @time BLAS.axpy!(2.0, V, V)
        @test A[Block(N,N)] ≈ 3AN

        A = BlockBandedMatrix(Ones{Float64}((4,6)), ([2,2], [2,2,2]), (0,2))
        B = BlockBandedMatrix(Ones{Float64}((6,6)), ([2,2,2], [2,2,2]), (0,1))
        @test sum(A) == 20
        @test sum(B) == 20
        C = BlockBandedMatrix{Float64}(undef, ([2,2], [2,2,2]), (0,3))
        @test all(copyto!(C, Mul(A,B)) .=== materialize!(MulAdd(1.0,A,B,0.0,similar(C))) .===
                    A*B)
        AB = A*B
        @test AB isa BlockBandedMatrix
        @test Matrix(AB) ≈ Matrix(A)*Matrix(B)
    end

    @testset "BlockBandedMatrix fill and copy" begin
        l , u = 1,1
        N = M = 10
        cols = rows = 1:N
        A = BlockBandedMatrix{Float64}(undef, (rows,cols), (l,u))
            A.data .= randn(length(A.data))

        fill!(view(A, Block(2,1)), 2.0)
        @test A[Block(2,1)] ≈ fill(2.0, 2)

        @test_throws BandError fill!(view(A, Block(3,1)), 2.0)

        fill!(view(A, Block(3,1)), 0.0)
        @test A[Block(3,1)] ≈ zeros(3)


        @test_throws BandError fill!(A, 2.0)
        fill!(A, 0.0)
        @test Matrix(A) == zeros(size(A))
    end

    @testset "BlockBandedMatrix type inferrence bug (#9)" begin
        s = BlockBandedMatrices.BlockBandedSizes([1, 2], [1, 2], 0, 0)
        f(s) = s.block_starts.data
        @inferred(f(s))
    end
end