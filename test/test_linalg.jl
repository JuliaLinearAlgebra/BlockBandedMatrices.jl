using BlockArrays, BlockBandedMatrices, Compat.Test
    import BandedMatrices: BandError
    import BlockBandedMatrices: _BandedBlockBandedMatrix, scalemul!, _scalemul!, memorylayout


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
