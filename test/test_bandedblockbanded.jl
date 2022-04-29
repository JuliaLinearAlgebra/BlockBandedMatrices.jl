using BlockArrays, BandedMatrices, BlockBandedMatrices, FillArrays, SparseArrays, Test, ArrayLayouts , LinearAlgebra
import BlockBandedMatrices: _BandedBlockBandedMatrix, blockcolsupport, blockrowsupport, colsupport, rowsupport,
                            isbandedblockbanded, bandeddata, BandedBlockBandedColumns
import ArrayLayouts: RangeCumsum

@testset "BandedBlockBandedMatrix" begin
    @testset "constructors" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 4
        cols = rows = 1:N

        @test Matrix(BandedBlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u), (λ,μ))) ==
            Array(BandedBlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u), (λ,μ))) ==
            zeros(Float64, 10, 10)

        @test Matrix(BandedBlockBandedMatrix{Int}(Zeros(sum(rows),sum(cols)), rows,cols, (l,u), (λ,μ))) ==
            zeros(Int, 10, 10)

        @test Matrix(BandedBlockBandedMatrix(Eye(sum(rows)), rows,cols, (l,u), (λ,μ))) ==
            Matrix{Float64}(I, 10, 10)

        @test Matrix(BandedBlockBandedMatrix{Int}(Eye(sum(rows)), rows,cols, (l,u), (λ,μ))) ==
            Matrix{Int}(I, 10, 10)

        @test Matrix(BandedBlockBandedMatrix(I, rows,cols, (l,u), (λ,μ))) ==
            Matrix{Float64}(I, 10, 10)

        @test Matrix(BandedBlockBandedMatrix{Int}(I, rows,cols, (l,u), (λ,μ))) ==
            Matrix{Int}(I, 10, 10)

        A = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20; 21 22 23 24 25]
        B = BandedBlockBandedMatrix(A, [2,3], [2,3], (0,1), (1,1))
        @test Matrix(B) ==
            [1 2 3 4 0; 6 7 8 9 10; 0 0 13 14 0; 0 0 18 19 20; 0 0 0 24 25]

        l,u,λ,μ = 0,0,4,0
        cols = 1:2:3
        rows = 2*cols
        A = BandedBlockBandedMatrix(fill(1.0,sum(rows),sum(cols)), rows,cols, (l,u), (λ,μ))
        B = [1.0 0.0 0.0 0.0;
             1.0 0.0 0.0 0.0;
             0.0 1.0 0.0 0.0;
             0.0 1.0 1.0 0.0;
             0.0 1.0 1.0 1.0;
             0.0 1.0 1.0 1.0;
             0.0 1.0 1.0 1.0;
             0.0 0.0 1.0 1.0]
        @test Matrix(A) == B

        @test Matrix(sparse(A)) == B
    end

    @testset "col and rowranges" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 4
        cols = rows = 1:N
        data = reshape(collect(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))
        @test Matrix(BlockBandedMatrix(A)) == Matrix(A)

        @test blockbandwidths(A) == (l,u)
        @test BlockBandedMatrices.subblockbandwidths(A) == (l,u)


        @test blockrowsupport(A, Block(1)) == Block.(1:2)
        @test blockrowsupport(A, Block(2)) == Block.(1:3)
        @test blockrowsupport(A, Block(3)) == Block.(2:4)

        @test blockcolsupport(A, Block(1)) == Block.(1:2)
        @test blockcolsupport(A, Block(2)) == Block.(1:3)
        @test blockcolsupport(A, Block(3)) == Block.(2:4)

        @test rowsupport(A,1) == 1:3
        @test rowsupport(A,2) == 1:6
        @test rowsupport(A,3) == 1:6
        @test rowsupport(A,4) == 2:10


        @test colsupport(A,1) == 1:3
        @test colsupport(A,2) == 1:6
        @test colsupport(A,3) == 1:6
        @test colsupport(A,4) == 2:10


        l , u = 2,1
        λ , μ = 1,2
        N = M = 4
        cols = rows = 1:N
        data = reshape(Vector(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))

        @test A.l == l == blockbandwidth(A,1)
        @test A.u == u == blockbandwidth(A,2)
        @test blockbandwidths(A) == (l, u)

        @test A.λ == λ == subblockbandwidth(A,1)
        @test A.μ == μ == subblockbandwidth(A,2)
        @test subblockbandwidths(A) == (λ, μ)

        @test blockrowsupport(A, Block(1)) == Block.(1:2)
        @test blockrowsupport(A, Block(2)) == Block.(1:3)
        @test blockrowsupport(A, Block(3)) == Block.(1:4)
        @test blockrowsupport(A, Block(4)) == Block.(2:4)


        @test blockcolsupport(A, Block(1)) == Block.(1:3)
        @test blockcolsupport(A, Block(2)) == Block.(1:4)
        @test blockcolsupport(A, Block(3)) == Block.(2:4)
        @test blockcolsupport(A, Block(4)) == Block.(3:4)

        @test rowsupport(A,1) == 1:3
        @test rowsupport(A,2) == 1:6
        @test rowsupport(A,3) == 1:6
        @test rowsupport(A,4) == 1:10

        @test colsupport(A,1) == 1:6
        @test colsupport(A,2) == 1:10
        @test colsupport(A,3) == 1:10
        @test colsupport(A,4) == 2:10

        l , u = -1,1
        λ , μ = 0,1
        rows = 1:5
        cols = 1:6

        data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), (λ+μ+1, (l+u+1)*sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))
        @test_throws BandError A[1,1] = 5

        @test A[1,2] == 4
        @test A[Block(2,2)] == [0 0; 0 0]
        @test A[Block(2,3)]  == [8.0 9.0 0.0; 0.0 10.0 11.0]
        @test bandwidths(view(A,Block(2,3))) == bandwidths(A[Block(2,3)]) == (0,1)

        @test blockrowsupport(A, Block(1)) == Block.(2:2)
        @test blockrowsupport(A, Block(2)) == Block.(3:3)
        @test blockrowsupport(A, Block(3)) == Block.(4:4)
        @test blockrowsupport(A, Block(4)) == Block.(5:5)

        @test blockcolsupport(A, Block(1)) == Block.(1:0)
        @test blockcolsupport(A, Block(2)) == Block.(1:1)
        @test blockcolsupport(A, Block(3)) == Block.(2:2)

        @test rowsupport(A,1) == 2:3
        @test rowsupport(A,2) == 4:6
        @test rowsupport(A,3) == 4:6
        @test rowsupport(A,4) == 7:10


        @test colsupport(A,1) == 1:0
        @test colsupport(A,2) == 1:1
        @test colsupport(A,3) == 1:1
        @test colsupport(A,4) == 2:3

        l , u = -1,1
        λ , μ = -1,1
        rows = 1:5
        cols = 1:6

        data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), (λ+μ+1, (l+u+1)*sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))
        @test_throws BandError A[1,1] = 5

        @test blockcolsupport(A, Block(1)) == Block.(1:0)
        @test blockcolsupport(A, Block(2)) == Block.(1:1)
        @test blockcolsupport(A, Block(3)) == Block.(2:2)

        @test colsupport(A,1) == 1:0
        @test colsupport(A,2) == 1:1
        @test colsupport(A,3) == 1:1
        @test colsupport(A,4) == 2:3
    end

    @testset "block indexing" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 4
        cols = rows = 1:N
        data = reshape(collect(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))

        @test A[Block(1), Block(1)] isa BandedMatrix
        @test A[Block(1), Block(1)] == A[Block(1,1)] == view(A, Block(1, 1)) == BandedMatrix(view(A, Block(1,1)))
        @test A[1,1] == view(A,Block(1),Block(1))[1,1] == view(A,Block(1,1))[1,1] == A[Block(1,1)][1,1]  == A[Block(1),Block(1)][1,1] == 5
        @test A[2,1] == view(A,Block(2),Block(1))[1,1] == view(A,Block(2,1))[1,1] == 8
        @test A[3,1] == view(A,Block(2),Block(1))[2,1] == 9
        @test A[4,1] == 0
        @test A[1,2] == view(A,Block(1,2))[1,1] == 11
        @test A[1,3] == view(A,Block(1,2))[1,2] == view(A,Block(1,2))[2] == 19

        @test bandwidths(view(A, Block(3),Block(1))) == (-720,-720)
        @test isempty(bandeddata(view(A, Block(3),Block(1))))

        @test A[Block(3,1)] == view(A, Block(3),Block(1)) == zeros(3,1)
        @test A[Block(3,1)] ≈ view(A, Block(3),Block(1)) ≈ zeros(3,1)
        @test_throws BandError view(A, Block(3),Block(1))[1,1] = 4
        @test_throws BlockBoundsError view(A, Block(5,1))
    end

    @testset "indexing" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 10
        cols = rows = 1:N
        data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1),sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))

        A[1,1] = 5
        @test A[1,1] == 5

        @test_throws BandError A[1,4] = 5
        A[1,4] = 0
        @test A[1,4] == 0

        # TODO: return a BandedMatrix
        @test A[1:10,1:10] ≈ Matrix(A)[1:10,1:10]
    end

    @testset "banded matrix interface for blocks" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 10
        cols = rows = 1:N
        data = reshape(collect(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))

        # test blocks
        V = view(A, Block(1,1))
        @test_throws BoundsError V[2,1]

        # BandedMatrix interface
        @test isbanded(V)
        @test bandwidths(V) == BlockBandedMatrices.subblockbandwidths(A)
        @test V[band(0)] ≈ view(V, band(0)) ≈ A[1:1,1:1]


        # test views of blocks fulfill BnadedMatrix interface
        @test BandedMatrices.inbands_getindex(V, 1, 1) == V[1,1] == 5
        BandedMatrices.inbands_setindex!(V, -1, 1, 1)
        @test A[1,1] == -1
        # these should throw errors but inbands turns it off

        V = view(A, Block(3,4))
        @test V[3,1] == 0
        @test_throws BandError V[3,1] = 5

        view(V, band(0)) .= -3
        @test all(A[Block(3,4)][band(0)] .== -3)

        @test BandedMatrix(V) isa BandedMatrix{Int,Matrix{Int}}
        @test BandedMatrix{Float64}(V) isa BandedMatrix{Float64,Matrix{Float64}}
        @test BandedMatrix{Float64}(BandedMatrix(V)) == BandedMatrix{Float64}(V) ==
                convert(BandedMatrix{Float64}, V)
        @test A[4:6,7:10] ≈ BandedMatrix(V)

        @test A[Block(3,4)].l == A.λ
        @test A[Block(3,4)].u == A.μ

        A[Block(3,4)] = BandedMatrix(Ones{Int}(3,4),(1,1))
        @test A[Block(3,4)] == BandedMatrix(Ones{Int}(3,4),(1,1))

        l , u = 2,1
        λ , μ = 1,2
        N = M = 4
        cols = rows = 1:N
        data = reshape(Vector(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))

        @test_throws BandError A[1,4] = 5
        @test_throws BandError view(A, Block(1,3))[2] = 5

        lu = (l , u) = -1,1
        λμ = (λ , μ) = -1,1
        rows = 1:5
        cols = 1:6

        data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))
        @test_throws BandError A[1,1] = 5

        @test A[1,3] == 3
        @test A[Block(2,2)] == [0 0; 0 0]
        @test A[Block(2,3)]  == [0 5 0; 0 0 6]
        @test bandwidths(A[Block(2,3)]) == (-1,1)
    end

    @testset "banded-block-banded matrix interface for blockranges" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 10
        cols = rows = 1:N
        data = reshape(collect(1.0:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))
        V = view(A, Block.(2:3), Block.(3:4))
        @test isbandedblockbanded(V)

        B = BandedBlockBandedMatrix(V)
        @test B isa BandedBlockBandedMatrix
        @test blockbandwidths(V) == blockbandwidths(B) == (2,0)
        @test subblockbandwidths(V) == subblockbandwidths(B) == subblockbandwidths(A) == (λ,μ)
        @test B == V == A[Block.(2:3), Block.(3:4)]

        @test A[Block.(2:3), Block.(3:4)] isa BandedBlockBandedMatrix

        x = randn(size(B,2))
        y = similar(x, size(B,1))
        @test all((similar(y) .= MulAdd(B, x)) .=== (similar(y) .= MulAdd(V,x)))
    end

    @testset "BLAS arithmetic" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 10
        cols = rows = fill(1000,N)
        data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))

        V = view(A, Block(N,N))

        AN = A[Block(N,N)]
        BLAS.axpy!(2.0, V, V)
        @test A[Block(N,N)] ≈ 3AN


        Y = zeros(cols[N], cols[N])
        BLAS.axpy!(2.0, V, Y)
        @test Y ≈ 2A[Block(N,N)]

        Y = BandedMatrix(Zeros(cols[N], cols[N]), (λ, μ))
        BLAS.axpy!(2.0, V, Y)
        @test Y ≈ 2A[Block(N,N)]

        Y = BandedMatrix(Zeros(cols[N], cols[N]), (λ+1, μ+1))
        BLAS.axpy!(2.0, V, Y)
        @test Y ≈ 2A[Block(N,N)]

        Y = BandedMatrix(Zeros(cols[N], cols[N]), (0, 0))
        @test_throws BandError BLAS.axpy!(2.0, V, Y)
    end

    @testset "Float32"  begin
        A = BandedBlockBandedMatrix{Float32}(Zeros{Float32}(10,10),
                                    fill(1,10), fill(1,10), (1,1), (1,1))

        @test eltype(A) == Float32


        A = BandedBlockBandedMatrix(Zeros{Float32}(10,10),
                                    fill(1,10), fill(1,10), (1,1), (1,1))

        @test eltype(A) == Float32
    end

    @testset "fill and copy" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 10
        cols = rows = 1:N
        data = randn((λ+μ+1)*(l+u+1), sum(cols))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))

        V = view(A,Block(2,3))
        @test_throws BandError fill!(V, 2.0)
        fill!(V, 0)
        @test A[2:3,4:6] == zeros(2,3)


        dataA = randn((λ+μ+1)*(l+u+1), sum(cols))
        A = _BandedBlockBandedMatrix(copy(dataA), rows,cols, (l,u), (λ,μ))

        K,J = 2,1
        V = view(A,Block(K),Block(J))
        fill!(V, 2.0)
        @test Matrix(V) == fill(2.0, 2,1)
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
        A = _BandedBlockBandedMatrix(copy(dataA), rows,cols, (l,u), (λ,μ))

        dataB = randn((λ+μ+3)*(l+u+3), sum(cols))
        B = _BandedBlockBandedMatrix(copy(dataB), rows,cols, (l+1,u+1), (λ+1,μ+1))


        B = _BandedBlockBandedMatrix(copy(dataB), rows,cols, (l+1,u+1), (λ+1,μ+1))
        copyto!(view(B, Block(N,N)), view(A, Block(N,N)))
        @test B[Block(N,N)] == A[Block(N,N)]

        B = _BandedBlockBandedMatrix(copy(dataB), rows,cols, (l+1,u+1), (λ+1,μ+1))
        copyto!(B, A)
        @test norm(B[Block(3,1)]) == 0
        @test  B ≈ A
        @test Matrix(B) ≈ Matrix(A)
    end


    @testset "Zero bands" begin
        B = BandedBlockBandedMatrix{Float64}(undef, 1:5,1:5, (-1,-1), (-1,-1))
        @test Matrix(B) == zeros(size(B))

        B = BandedBlockBandedMatrix{Float64}(undef, 1:5,1:5, (-1,-1), (1,-1))
        @test Matrix(B) == zeros(size(B))

        B = BandedBlockBandedMatrix{Float64}(undef, 1:5,1:5, (1,-1), (-1,-1))
        @test Matrix(B) == zeros(size(B))
    end

    @testset "BandedBlockBanded with BlockMatrix" begin
        WithBlockMatrix{T} = BandedBlockBandedMatrix{T, BlockMatrix{T, Matrix{Matrix{T}}}}
        args = [1, 2, 3], [2, 2, 1], (1, 1), (1, 1)
        A = WithBlockMatrix{Int64}(undef, args...)
        B = BandedBlockBandedMatrix{Int64}(undef, axes(A), blockbandwidths(A), subblockbandwidths(A))

        @test eltype(A) === eltype(B) === Int64
        @test typeof(A.data) <: BlockArray
        @test typeof(B.data) <: PseudoBlockArray
        @test size(A) == size(B)
        @test bandrange(A) == bandrange(B)
        @test blockbandwidths(A) == blockbandwidths(B)
        @test blocksize(A) == blocksize(B)

        A = WithBlockMatrix{Int64}(Zeros{Int64}(sum(args[1]),sum(args[2])), args...)
        B = BandedBlockBandedMatrix{Int64}(Zeros{Int64}(sum(args[1]),sum(args[2])), args...)
        @test typeof(A.data) <: BlockArray
        @test typeof(B.data) <: PseudoBlockArray
        @test A == B

        A = WithBlockMatrix{Int64}(Ones{Int64}(sum(args[1]),sum(args[2])), args...)
        B = BandedBlockBandedMatrix{Int64}(Ones{Int64}(sum(args[1]),sum(args[2])), args...)
        @test typeof(A.data) <: BlockArray
        @test typeof(B.data) <: PseudoBlockArray
        @test A == B
        @test (A .+ 1) .* 2 == B .* 2 .+ 2
    end

    @testset "Sparse dimensions" begin
        A = BandedBlockBandedMatrix{Float64}(undef, 1:5,1:5, (-1,1), (-1,1))
        @test size(sparse(A)) == size(A) == (15,15)
    end

    @testset "non-standard blocks" begin
        A = BandedBlockBandedMatrix{Float64}(undef, Int[], 1:5,(-1,1), (-1,1))
        @test BlockBandedMatrices.colstart(A,1) == 1
        A = BandedBlockBandedMatrix{Float64}(undef, 1:2, 1:5,(-1,1), (-1,1))
        A.data .= randn.()
        V = view(A, Block(2,3))
        @test MemoryLayout(typeof(V)) == BandedMatrices.BandedColumns{ColumnMajor}()
        @test isbanded(V)
        @test bandwidths(V) == (-1,1)
        @test BandedMatrix(V) == A[2:3,4:6]
        V2 = view(V, :, 2:3)
        @test MemoryLayout(typeof(V2)) == BandedMatrices.BandedColumns{ColumnMajor}()
        @test bandwidths(V2) == (0,0)
        @test BandedMatrix(V2) == A[2:3,5:6]
        V2 = view(V, 2:2, :)
        @test MemoryLayout(typeof(V2)) == BandedMatrices.BandedColumns{ColumnMajor}()
        @test bandwidths(V2) == (-2,2)
        @test BandedMatrix(V2) == A[3:3,4:6]
        V2 = view(V, 2:2, 2:3)
        @test MemoryLayout(typeof(V2)) == BandedMatrices.BandedColumns{ColumnMajor}()
        @test bandwidths(V2) == (-1,1)
        @test BandedMatrix(V2) == A[3:3,5:6]
    end

    @testset "OneTo axes" begin
        ret = BandedBlockBandedMatrix{Int}(undef, (Base.OneTo(5),Base.OneTo(5)), (0,0), (1,1))
        V = view(ret, Block(1,1))
        @test bandwidths(V) == (1,1)
        A = brand(5,5,1,1)
        B = BandedBlockBandedMatrix(A)
        @test A == B
        @test subblockbandwidths(B) == (1,1)
        @test blockbandwidths(B) == (0,0)
    end

    @testset "DualLayout blocks" begin
        A = _BandedBlockBandedMatrix(PseudoBlockVector([1,2,3],[1,2])', blockedrange([1,2]), (-1,1), (-1,1))
        @test MemoryLayout(A) isa BandedBlockBandedColumns{RowMajor}
    end

    @testset "1:N blocks" begin
        N = 10
        A = BandedBlockBandedMatrix{Float64}(undef, 1:N,1:N, (1,1), (1,1))
        @test axes(A) isa NTuple{2,BlockedUnitRange{<:RangeCumsum}}
    end

    @testset "change bandwidths" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 4
        cols = rows = 1:N
        data = reshape(collect(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))
        @test A == BandedBlockBandedMatrix(A, (2,1), (2,1)) == BandedBlockBandedMatrix{Float64}(A, (2,1), (2,1))
    end

    @testset "convert" begin
        l , u = 1,1
        λ , μ = 1,1
        N = M = 4
        cols = rows = 1:N
        data = reshape(collect(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows,cols, (l,u), (λ,μ))
        @test AbstractArray{Float64}(A) == AbstractMatrix{Float64}(A) == AbstractArray{Int}(A) == AbstractMatrix{Int}(A) == copy(A) == A
        @test convert(AbstractArray{Float64}, A) == convert(AbstractMatrix{Float64}, A) == A
        @test convert(AbstractArray{Int}, A) ≡ convert(AbstractMatrix{Int}, A) ≡ A
    end

    @testset "Symmetric" begin
        l, u = 2, 1
        λ, μ = 2, 1
        N = M = 4
        cols = rows = 1:N
        data = reshape(collect(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ + μ + 1) * (l + u + 1), sum(cols)))
        A = _BandedBlockBandedMatrix(data, rows, cols, (l, u), (λ, μ))
        
        @test blockbandwidths(Hermitian(A)) == blockbandwidths(Symmetric(A)) == (1,1)
        @test blockbandwidths(Hermitian(A,:L)) == blockbandwidths(Symmetric(A,:L)) == (2,2)
        @test subblockbandwidths(Hermitian(A)) == subblockbandwidths(Symmetric(A)) == (2,2)
        @test subblockbandwidths(Hermitian(A,:L)) == subblockbandwidths(Symmetric(A,:L)) == (2,2)

        @test BandedBlockBandedMatrix(Symmetric(A)) == Symmetric(Matrix(A))
        @test BandedBlockBandedMatrix(Symmetric(A,:L)) == Symmetric(Matrix(A),:L)
        @test BandedBlockBandedMatrix(Hermitian(A)) == Hermitian(Matrix(A))
        @test BandedBlockBandedMatrix(Hermitian(A,:L)) == Hermitian(Matrix(A),:L)

        @test Symmetric(A)[Block.(1:3),Block.(1:3)] isa BandedBlockBandedMatrix
        @test Hermitian(A)[Block.(1:3),Block.(1:3)] isa BandedBlockBandedMatrix
    end
end

if false # turned off since tests have check-bounds=yes
    # test that @inbounds is working properly
    exceed_band(V, k, j) = @inbounds return V[k,j]
    @test exceed_band(V, 2,1) == 8

    @test BandedMatrices.inbands_getindex(V, 2, 1) == 8
    BandedMatrices.inbands_setindex!(V, -2, 5, 1)
    @test A[2,1] == -2
end
