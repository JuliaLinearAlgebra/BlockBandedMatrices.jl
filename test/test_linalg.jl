using BlockArrays, BlockBandedMatrices, Compat.Test
    import BandedMatrices: BandError
    import BlockBandedMatrices: _BandedBlockBandedMatrix, MemoryLayout, mul!,
                                blockcolstop, blockrowstop, BlockSizes, block_sizes




@testset "BlockBandedMatrix linear algebra" begin
    l , u = 1,1
    N = M = 4
    cols = rows = 1:N
    A = BlockBandedMatrix{Float64}(uninitialized, (rows,cols), (l,u))
        A.data .= 1:length(A.data)

    V = view(A, Block(N), Block(N))

    @test strides(V) == (1,7)
    @test stride(V,2) == 7
    @test unsafe_load(pointer(V)) == 46
    @test unsafe_load(pointer(V) + stride(V,2)*sizeof(Float64)) == 53

    v = ones(4)
    U = UpperTriangular(view(A, Block(N), Block(N)))
    w = Matrix(U) \ v
    U \ v == w
    @test v == ones(4)
    @test A_ldiv_B!(U , v) === v
    @test v == w

    v = ones(size(A,1))

    U = UpperTriangular(A)
    w = Matrix(U) \ v
    @test U \ v ≈ w

    @test v == ones(size(A,1))
    @test A_ldiv_B!(U, v) === v
    @test v ≈ w
end

@testset "BandedBlockBandedMatrix linear algebra" begin
    l , u = 1,1
    λ , μ = 1,1
    N = M = 10
    cols = rows = 1:N

    data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
    A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))

    V = view(A, Block(2), Block(2))
    @test unsafe_load(Base.unsafe_convert(Ptr{Float64}, V)) == 13.0




    @which Base.unsafe_convert(Ptr{Float64}, V)

    C =  A*A
    @test C isa BandedBlockBandedMatrix
    @test Matrix(A*A) ≈ Matrix(A)*Matrix(A)
    @test C.l == C.u == C.λ == C.μ == 2



    A = BlockBandedMatrix{Float64}(uninitialized, (rows,cols), (l,u))
        A.data .= 1:length(A.data)

    V = view(A, Block(2,2))

    W = 2.0Matrix(V)^2 + 3.0Matrix(V)
    C = copy(V)
    BLAS.gemm!('N', 'N', 2.0, V, V, 3.0, C)
    @test C == W
    BLAS.gemm!('N', 'N', 2.0, V, V, 3.0, V)
    @test V == W

    BLAS.gemm!('N', 'N', 2.0, ones(V), V, 0.0, C)
    @test 2.0*ones(V)*V == C

    BLAS.gemm!('N', 'N', 2.0, V, ones(V), 0.0, C)
    @test 2.0*V*ones(V) == C



    l , u = 1,1
    λ , μ = 1,1
    N = M = 20
    cols = rows = 1:N

    dataA = randn((λ+μ+1)*(l+u+1), sum(cols))
    A = _BandedBlockBandedMatrix(copy(dataA), (rows,cols), (l,u), (λ,μ))

    K,J = 2,1
    @test_throws BandError fill!(view(A,Block(K),Block(J)), 2.0)
    fill!(view(A,Block(K),Block(J)), 0.0)
    @test Matrix(view(A,Block(K),Block(J))) == zeros(2,1)

    K,J = 3,1
    @test_throws BandError fill!(view(A,Block(K),Block(J)), 2.0)
    fill!(view(A,Block(K),Block(J)), 0.0)
    @test Matrix(view(A,Block(K),Block(J))) == zeros(3,1)

    @test_throws BandError fill!(A, 2.0)
    fill!(A, 0.0)
    @test Matrix(A) == zeros(size(A))

    dataA = randn((λ+μ+1)*(l+u+1), sum(cols))
    A = _BandedBlockBandedMatrix(copy(dataA), (rows,cols), (l,u), (λ,μ))

    dataB = randn((λ+μ+3)*(l+u+3), sum(cols))
    B = _BandedBlockBandedMatrix(copy(dataB), (rows,cols), (l+1,u+1), (λ+1,μ+1))

    @time AB = A + B
    @test AB ≈ Matrix(A) + Matrix(B)

    @time BLAS.axpy!(1.0, A, B)
    @test B ≈ AB
end



@testset "SubBlockBandedMatrix linear algebra" begin
    l , u = 1,1
    N = M = 5
    cols = rows = 1:N
    A = BlockBandedMatrix{Float64}(uninitialized, (rows,cols), (l,u))
        A.data .= randn(length(A.data))



    V = view(A, Block.(1:3), Block.(1:3))

    @test blockrowstop(V,1) == Block(2)
    @test blockcolstop(V,1) == Block(2)

    @test BlockBandedMatrices.block_sizes(V) == BlockSizes(1:3, 1:3)

    b = randn(size(V,1))
    r = UpperTriangular(Matrix(V)) \ b
    @test BlockBandedMatrices.blockbanded_squareblocks_trtrs!(V, copy(b)) ≈ r

    @test all(A_ldiv_B!(UpperTriangular(V), copy(b)) .=== BlockBandedMatrices.blockbanded_squareblocks_trtrs!(V, copy(b)))

    V = view(A, Block.(2:3), Block(3))
    @test unsafe_load(pointer(V)) == A[2,4]
    @test unsafe_load(pointer(V)+sizeof(Float64)*stride(V,2)) == A[2,5]
    @test MemoryLayout(V) == BandedMatrices.ColumnMajor{Float64}()

    @test size(V) == (5,3)
    b = randn(size(V,2))
    @test all(V*b .=== Matrix(V)*b .=== BlockBandedMatrices.gemv!('N', 1.0, V, b, 0.0, Vector{Float64}(uninitialized, size(V,1))))

    V = view(A, Block.(1:3), Block(3)[2:3])
    @test_throws ArgumentError pointer(V)
    @test_throws ArgumentError MemoryLayout(V)

    b = randn(size(A,1))
    @test UpperTriangular(A) \ b ≈ UpperTriangular(Matrix(A)) \ b

    V = view(A, Block.(2:3), Block(3)[2:3])
    @test unsafe_load(pointer(V)) == A[2,5]
    @test unsafe_load(pointer(V)+sizeof(Float64)*stride(V,2)) == A[2,6]
    @test MemoryLayout(V) == BandedMatrices.ColumnMajor{Float64}()

    @test size(V) == (5,2)
    b = randn(size(V,2))
    @test all(V*b .=== Matrix(V)*b .=== BlockBandedMatrices.gemv!('N', 1.0, V, b, 0.0, Vector{Float64}(uninitialized, size(V,1))))

    V = view(A, Block.(1:3), Block(3)[2:3])
    @test_throws ArgumentError pointer(V)
    @test_throws ArgumentError MemoryLayout(V)

    b = randn(size(A,1))
    @test UpperTriangular(A) \ b ≈ UpperTriangular(Matrix(A)) \ b

    V_22 = view(A, Block(N)[1:N],  Block(N)[1:N])
    @test MemoryLayout(V_22) == BandedMatrices.ColumnMajor{Float64}()

    V = view(A, Block(N),  Block(N))
    V_22 = view(A, Block(N)[1:N],  Block(N)[1:N])
    @test unsafe_load(pointer(V_22)) == V_22[1,1] == V[1,1]
    @test strides(V_22) == strides(V) == (1,9)
    b = randn(N)
    @test all(V*b .=== V_22*b .=== Matrix(V)*b .===
        BlockBandedMatrices.gemv!('N', 1.0, V, b, 0.0, Vector{Float64}(uninitialized, size(V,1))))

    @test all(UpperTriangular(V_22) \ b .=== A_ldiv_B!(UpperTriangular(V_22) , copy(b)) .=== A_ldiv_B!(UpperTriangular(V) , copy(b)) .===
        A_ldiv_B!(UpperTriangular(Matrix(V_22)) , copy(b)))


    V = view(A, Block.(rows), Block.(cols))
    V2 = view(A, 1:size(A,1), 1:size(A,2))
    b = randn(size(V,1))

    @test all(Matrix(V) .=== Matrix(V2))
    UpperTriangular(V2) \ b ≈ UpperTriangular(V) \ b
end

@testset "Rectangular blocks BlockBandedMatrix linear algebra" begin
    l , u = 0,1
    rows = [3,4,5]
    cols = [2,3,4,3]

    A = BlockBandedMatrix{Float64}(uninitialized, (rows,cols), (l,u))
        A.data .= randn(length(A.data))

    b = randn(size(A,1))

    V = view(A, Block.(1:3), Block.(1:4))
    @test block_sizes(V) == block_sizes(A)
    @test all(Matrix(V) .=== Matrix(A))

    @test view(V, Block(2)[1:2], Block(3)) ≡ view(A, Block(2)[1:2], Block(3))

    @test all(UpperTriangular(A) \ b .===
                    BlockBandedMatrices.blockbanded_rectblocks_trtrs!(A, copy(b)) .===
                    BlockBandedMatrices.blockbanded_rectblocks_trtrs!(V, copy(b)))
    @test UpperTriangular(A) \ b ≈ UpperTriangular(V) \ b ≈ UpperTriangular(Matrix(A)) \ b

    V = view(A, 1:11, 1:11)
    b = randn(size(V,1))
    @test all(UpperTriangular(V) \ b .===
                    BlockBandedMatrices.blockbanded_rectblocks_intrange_trtrs!(V, copy(b)))
    @test UpperTriangular(V) \ b ≈ UpperTriangular(Matrix(V)) \ b ≈ UpperTriangular(A[1:11,1:11]) \ b
end
