using BlockArrays, BandedMatrices, BlockBandedMatrices, LazyArrays, LinearAlgebra, Test
    import BandedMatrices: BandError, bandeddata
    import BlockBandedMatrices: _BandedBlockBandedMatrix, MemoryLayout, mul!,
                                blockcolstop, blockrowstop, BlockSizes, blocksizes,
                                BlockSkylineSizes
    import LazyArrays: ColumnMajor



@testset "lmul!/rmul!" begin
    C = BandedBlockBandedMatrix{Float64}(undef, (1:2,1:2), (1,1), (1,1))
    C.data .= NaN
    lmul!(0.0, C)
    norm(C) == 0.0

    C.data .= NaN
    rmul!(C, 0.0)
    norm(C) == 0.0
end

@testset "BlockBandedMatrix linear algebra" begin
    l , u = 1,1
    N = M = 4
    cols = rows = 1:N
    A = BlockBandedMatrix{Float64}(undef, (rows,cols), (l,u))
        A.data .= 1:length(A.data)

    V = view(A, Block(N,N))

    @test strides(V) == (1,7)
    @test stride(V,2) == 7
    @test unsafe_load(pointer(V)) == 46
    @test unsafe_load(pointer(V) + stride(V,2)*sizeof(Float64)) == 53

    x = randn(size(A,2))
        @test A*x == (similar(x) .= Mul(A,x)) ≈ Matrix(A)*x

    z = randn(size(A,2)) + im*randn(size(A,2))
    A*z == (similar(z) .= Mul(A,z)) ≈ Matrix(A)*z

    Matrix(A)*z
    z[1]
    view(A,Block(1,1)) * z[1] + view(A,Block(1,2))*z[2:3]

    A*z

    X = randn(size(A))
        @test A*X == (similar(X) .= Mul(A,X)) ≈ Matrix(A)*X
        @test X*A == (similar(X) .= Mul(X,A)) ≈  Matrix(X)*A


    Z = randn(size(A)) + im*randn(size(A))

    A*Z


    v = fill(1.0,4)
    U = UpperTriangular(view(A, Block(N,N)))
    @test Matrix(U) == U
    w = Matrix(U) \ v
    @test U \ v ≈ w
    @test v == fill(1.0,4)
    @test ldiv!(U , v) === v
    @test v ≈ w

    v = fill(1.0,size(A,1))

    U = UpperTriangular(A)
    w = Matrix(U) \ v
    @test U \ v ≈ w

    @test v == fill(1.0,size(A,1))
    @test ldiv!(U, v) === v
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
    @test unsafe_load(Base.unsafe_convert(Ptr{Float64}, bandeddata(V))) == 13.0

    C = BandedMatrix{Float64}(undef, size(V), 2 .*bandwidths(V))
    C .= Mul(V,V)
    @test all(C .=== BandedMatrix(V)*BandedMatrix(V))
    @test all((similar(C) .= 2.0 .* Mul(V,V) .+ C) .=== BandedMatrices.gbmm!('N', 'N', 2.0, V, V, 1.0, deepcopy(C)))

    C = BandedBlockBandedMatrix{Float64}(undef, (rows,cols), (2l,2u), (2λ,2μ))
    C .= Mul(A,A)

    @test Matrix(A) == [A[k,j] for k=1:size(A,1), j=1:size(A,2)]
    @test Matrix(C) ≈ Matrix(A)*Matrix(A)
    C =  A*A
    @test C isa BandedBlockBandedMatrix
    @test Matrix(C) ≈ Matrix(A)*Matrix(A)
    @test C.l == C.u == C.λ == C.μ == 2

    A = BlockBandedMatrix{Float64}(undef, (rows,cols), (l,u))
        A.data .= 1:length(A.data)

    V = view(A, Block(2,2))

    W = 2.0Matrix(V)^2 + 3.0Matrix(V)
    C = copy(V)
    BLAS.gemm!('N', 'N', 2.0, V, V, 3.0, C)
    @test C == W

    BLAS.gemm!('N', 'N', 2.0, fill!(similar(V),1.0), V, 0.0, C)
    @test 2.0*fill!(similar(V),1.0)*V == C

    BLAS.gemm!('N', 'N', 2.0, V, fill!(similar(V),1.0), 0.0, C)
    @test 2.0*V*fill!(similar(V),1.0) == C


    l , u = 1,1
    λ , μ = 1,1
    N = M = 20
    cols = rows = 1:N

    dataA = randn((λ+μ+1)*(l+u+1), sum(cols))
    A = _BandedBlockBandedMatrix(copy(dataA), (rows,cols), (l,u), (λ,μ))

    K,J = 2,1
    fill!(view(A,Block(K),Block(J)), 2.0)
    @test view(A,Block(K),Block(J)) == fill(2.0, 2, 1)
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

@testset "Rectangular block *" begin
    A = BlockBandedMatrix{Float64}(undef, (Ones{Int}(2), Ones{Int}(2)), (0,2))
        A.data .= randn.()
    B = BlockBandedMatrix{Float64}(undef, (Ones{Int}(2), Ones{Int}(3)), (0,2))
        B.data .= randn.()

    @test A*B ≈ Matrix(A)*Matrix(B)


    rows = rand(1:10, 5)
    l = rand(0:2, 5)
    u = rand(0:2, 5)

    m = sum(rows)

    A = BlockSkylineMatrix(Zeros(m,m), (rows,rows), (l,u))
    A.data .= randn.()

    B = BlockSkylineMatrix(Zeros(m,m+1), (rows,[rows;1]), ([l;1],[u;1]))
    B.data .= randn.()
    @test A*B ≈ Matrix(A)*Matrix(B)
end
