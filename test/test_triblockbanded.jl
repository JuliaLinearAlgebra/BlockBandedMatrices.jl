using BlockBandedMatrices, BandedMatrices, BlockArrays, LazyArrays, LinearAlgebra, Test
import BlockBandedMatrices: MemoryLayout, TriangularLayout, BandedBlockBandedColumnMajor,
                        BandedColumnMajor, tribandeddata, blocksizes, cumulsizes, nblocks,
                        BlockSkylineSizes, blockrowstop, blockcolstop, BlockSizes,
                        ColumnMajor, BlockBandedSizes

@testset "triangular" begin
    @testset "triangular BandedBlockBandedMatrix mul" begin
        A = BandedBlockBandedMatrix{Float64}(undef, (1:10,1:10), (1,1), (1,1))
            A.data .= randn.()

        U = UpperTriangular(A)
        @test MemoryLayout(typeof(U)) == TriangularLayout{'U','N',BandedBlockBandedColumnMajor}()
        b = randn(size(U,1))
        @test U*b isa Vector{Float64}
        @test (similar(b) .= Mul(U,b)) ≈ U*b  ≈ Matrix(U)*b


        U = UnitUpperTriangular(A)
        @test MemoryLayout(typeof(U)) == TriangularLayout{'U','U',BandedBlockBandedColumnMajor}()
        b = randn(size(U,1))
        @test U*b isa Vector{Float64}
        @test (similar(b) .= Mul(U,b)) ≈ U*b  ≈ Matrix(U)*b

        L = LowerTriangular(A)
        @test MemoryLayout(typeof(L)) == TriangularLayout{'L','N',BandedBlockBandedColumnMajor}()
        b = randn(size(U,1))
        @test L*b isa Vector{Float64}
        @test (similar(b) .= Mul(L,b)) ≈ L*b  ≈ Matrix(L)*b


        L = UnitLowerTriangular(A)
        @test MemoryLayout(typeof(L)) == TriangularLayout{'L','U',BandedBlockBandedColumnMajor}()
        b = randn(size(L,1))
        @test L*b isa Vector{Float64}
        @test (similar(b) .= Mul(L,b)) ≈ L*b  ≈ Matrix(L)*b
    end

    @testset "Block by BlockIndex" begin
        A = BandedBlockBandedMatrix{Float64}(undef, (1:10,1:10), (1,1), (1,1))
            A.data .= randn.()
        V = view(A, Block(3), Block.(3:5))
        @test MemoryLayout(typeof(V)) == BandedBlockBandedColumnMajor()
        b = randn(size(V,2))
        c = similar(b, size(V,1))

        @test blockbandwidths(V) == (1,1)
        @test (c .= Mul(V,b)) ≈ (Matrix(V)*b)

        V = view(A, Block.(2:4), Block(3))

        @test MemoryLayout(typeof(V)) == BandedBlockBandedColumnMajor()
        @test blockbandwidths(V) == (2,0)
        @test cumulsizes(blocksizes(V)) == ([1,3,6,10], [1,4])

        @test nblocks(V) == (3,1)
        V2 = view(V, Block(1), Block(1))
        @test MemoryLayout(typeof(V2)) == BandedColumnMajor()

        b = rand(size(V,2))
        @test (similar(b, size(V,1)) .= Mul(V, b)) ≈ V*b
    end

    @testset "triangular BandedBlockBandedMatrix ldiv" begin
        A = BandedBlockBandedMatrix{Float64}(undef, (1:10,1:10), (1,1), (1,1))
            A.data .= randn.()
            A

        U = UpperTriangular(A)
        b = randn(size(U,1))
        @test copyto!(similar(b), Ldiv(U,b)) ≈ (Matrix(U) \ b)
        @test all((similar(b) .= Ldiv(U, b)) .=== copyto!(similar(b), Ldiv(U,b)))

        U = UnitUpperTriangular(A)
        b = randn(size(U,1))
        @test copyto!(similar(b), Ldiv(U,b)) ≈ (Matrix(U) \ b)
        @test all((similar(b) .= Ldiv(U, b)) .=== copyto!(similar(b), Ldiv(U,b)))

        L = LowerTriangular(A)
        b = randn(size(L,1))
        @test copyto!(similar(b), Ldiv(L,b)) ≈ (Matrix(L) \ b)
        @test all((similar(b) .= Ldiv(L, b)) .=== copyto!(similar(b), Ldiv(L,b)))
    end


    @testset "Rectangular blocks BlockBandedMatrix linear algebra" begin
        l , u = 0,1
        rows = [3,4,5]
        cols = [2,3,4,3]

        A = BlockBandedMatrix{Float64}(undef, (rows,cols), (l,u))
            A.data .= randn(length(A.data))

        b = randn(size(A,1))

        V = view(A, Block.(1:3), Block.(1:4))
        @test blocksizes(V) isa BlockSkylineSizes
        @test blocksizes(V) == blocksizes(A)


        @test all(Matrix(V) .=== Matrix(A))

        @test view(V, Block(2)[1:2], Block(3)) ≡ view(A, Block(2)[1:2], Block(3))

        @test_broken all(UpperTriangular(A) \ b .===
                        BlockBandedMatrices.blockbanded_rectblocks_trtrs!(A, copy(b)) .===
                        BlockBandedMatrices.blockbanded_rectblocks_trtrs!(V, copy(b)))
        @test UpperTriangular(A) \ b ≈ UpperTriangular(V) \ b ≈ UpperTriangular(Matrix(A)) \ b

        V = view(A, 1:11, 1:11)
        b = randn(size(V,1))
        @test_skip all(UpperTriangular(V) \ b .===
                        BlockBandedMatrices.blockbanded_rectblocks_intrange_trtrs!(V, copy(b)))
        @test UpperTriangular(V) \ b ≈ UpperTriangular(Matrix(V)) \ b ≈ UpperTriangular(A[1:11,1:11]) \ b

        # bug from SingularIntegralEquations, fixed by k_old check in _squaredblocks_newbandwidth
        A = BlockBandedMatrix{Float64}(undef, ([5; fill(4,470)], [7; fill(4,368)]), (102,269))
            A.data .= randn(length(A.data))
            for k = 1:min(size(A,1), size(A,2))
                A[k,k] += 10
            end

        V = view(A, 1:400,1:400)
        b = randn(400)
        @test_broken all(UpperTriangular(V) \ b .=== BlockBandedMatrices.blockbanded_rectblocks_intrange_trtrs!(V, copy(b)))
        @test UpperTriangular(V) \ b ≈ UpperTriangular(Matrix(V)) \ b
    end

    @testset "SubBlockSkylineMatrix linear algebra" begin
        l , u = 1,1
        N = M = 5
        cols = rows = 1:N
        A = BlockBandedMatrix{Float64}(undef, (rows,cols), (l,u))
            A.data .= randn(length(A.data))

        V = view(A, Block.(1:3), Block.(1:3))

        @test blockrowstop(V,1) == Block(2)
        @test blockcolstop(V,1) == Block(2)

        @test blocksizes(V) == BlockBandedSizes(BlockSizes(1:3, 1:3), l,u)

        b = randn(size(V,1))
        r = UpperTriangular(Matrix(V)) \ b
        @test_broken BlockBandedMatrices.blockbanded_squareblocks_trtrs!(V, copy(b)) ≈ r

        @test_broken all(ldiv!(UpperTriangular(V), copy(b)) .=== BlockBandedMatrices.blockbanded_squareblocks_trtrs!(V, copy(b)))

        V = view(A, Block.(2:3), Block(3))
        @test unsafe_load(pointer(V)) == A[2,4]
        @test unsafe_load(pointer(V)+sizeof(Float64)*stride(V,2)) == A[2,5]
        @test MemoryLayout(typeof(V)) == ColumnMajor()

        @test size(V) == (5,3)
        b = randn(size(V,2))

        @test all((similar(b,size(V,1)) .= Mul(V,b)) .=== Matrix(V)*b .===
                    BLAS.gemv!('N', 1.0, V, b, 0.0, Vector{Float64}(undef, size(V,1))))

        V = view(A, Block.(1:3), Block(3)[2:3])
        @test_throws ArgumentError pointer(V)
        @test MemoryLayout(typeof(V)) == ColumnMajor()

        b = randn(size(A,1))
        @test UpperTriangular(A) \ b ≈ UpperTriangular(Matrix(A)) \ b

        V = view(A, Block.(2:3), Block(3)[2:3])
        @test unsafe_load(pointer(V)) == A[2,5]
        @test unsafe_load(pointer(V)+sizeof(Float64)*stride(V,2)) == A[2,6]
        @test MemoryLayout(typeof(V)) == ColumnMajor()

        @test size(V) == (5,2)
        b = randn(size(V,2))
        @test all((similar(b,size(V,1)) .= Mul(V,b)) .=== Matrix(V)*b .===
                    BLAS.gemv!('N', 1.0, V, b, 0.0, Vector{Float64}(undef, size(V,1))))

        V = view(A, Block.(1:3), Block(3)[2:3])
        @test_throws ArgumentError pointer(V)
        @test MemoryLayout(typeof(V)) == ColumnMajor()

        b = randn(size(A,1))
        @test UpperTriangular(A) \ b ≈ UpperTriangular(Matrix(A)) \ b

        V_22 = view(A, Block(N)[1:N],  Block(N)[1:N])
        @test MemoryLayout(typeof(V_22)) == ColumnMajor()

        V = view(A, Block(N),  Block(N))
        V_22 = view(A, Block(N)[1:N],  Block(N)[1:N])
        @test unsafe_load(pointer(V_22)) == V_22[1,1] == V[1,1]
        @test strides(V_22) == strides(V) == (1,9)
        b = randn(N)
        @test all(copyto!(similar(b) , Mul(V,b)) .=== copyto!(similar(b) , Mul(V_22,b)) .===
            Matrix(V)*b .===
            BLAS.gemv!('N', 1.0, V, b, 0.0, Vector{Float64}(undef, size(V,1))))

        @test_skip all(UpperTriangular(V_22) \ b .=== ldiv!(UpperTriangular(V_22) , copy(b)) .=== ldiv!(UpperTriangular(V) , copy(b)) .===
            ldiv!(UpperTriangular(Matrix(V_22)) , copy(b)))

        V = view(A, Block.(rows), Block.(cols))
        V2 = view(A, 1:size(A,1), 1:size(A,2))
        b = randn(size(V,1))

        @test all(Matrix(V) .=== Matrix(V2))
        UpperTriangular(V2) \ b ≈ UpperTriangular(V) \ b
    end
end