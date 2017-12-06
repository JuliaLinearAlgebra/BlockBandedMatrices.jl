using BlockArrays, BandedMatrices, BlockBandedMatrices, Base.Test
    import BlockBandedMatrices: _BandedBlockBandedMatrix

l , u = 1,1
λ , μ = 1,1
N = M = 4
cols = rows = 1:N

@test Matrix(BandedBlockBandedMatrix(Zeros(sum(rows),sum(cols)), (rows,cols), (l,u), (λ,μ))) ==
    zeros(Float64, 10, 10)

@test Matrix(BandedBlockBandedMatrix{Int}(Zeros(sum(rows),sum(cols)), (rows,cols), (l,u), (λ,μ))) ==
    zeros(Int, 10, 10)

@test Matrix(BandedBlockBandedMatrix(Eye(sum(rows),sum(cols)), (rows,cols), (l,u), (λ,μ))) ==
    eye(Float64, 10, 10)

@test Matrix(BandedBlockBandedMatrix{Int}(Eye(sum(rows),sum(cols)), (rows,cols), (l,u), (λ,μ))) ==
    eye(Int, 10, 10)

@test Matrix(BandedBlockBandedMatrix(I, (rows,cols), (l,u), (λ,μ))) ==
    eye(Float64, 10, 10)

@test Matrix(BandedBlockBandedMatrix{Int}(I, (rows,cols), (l,u), (λ,μ))) ==
    eye(Int, 10, 10)



data = reshape(collect(1:(λ+μ+1)*(l+u+1)*sum(cols)), (λ+μ+1, (l+u+1)*sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))

@test blockbandwidths(A) == (l,u)
@test BlockBandedMatrices.subblockbandwidths(A) == (l,u)

# The first block is ignored
BlockBandedMatrices.bbb_data_cols(view(A, Block(1,1))) == 2:2
BlockBandedMatrices.bbb_data_cols(view(A, Block(2,1))) == 3:3
BlockBandedMatrices.bbb_data_cols(view(A, Block(1,2))) == 4:5
BlockBandedMatrices.bbb_data_cols(view(A, Block(2,2))) == 6:7

# check views of blocks are indexing correctly

@test A[Block(1), Block(1)] isa BandedMatrix
@test A[Block(1), Block(1)] == A[Block(1,1)] == BlockArrays.getblock(A, 1, 1) == BandedMatrix(view(A, Block(1,1)))
@test A[1,1] == view(A,Block(1),Block(1))[1,1] == view(A,Block(1,1))[1,1] == A[Block(1,1)][1,1]  == A[Block(1),Block(1)][1,1] == 5
@test A[2,1] == view(A,Block(2),Block(1))[1,1] == view(A,Block(2,1))[1,1] == 8
@test A[3,1] == view(A,Block(2),Block(1))[2,1] == 9
@test A[4,1] == 0
@test A[1,2] == view(A,Block(1,2))[1,1] == 11
@test A[1,3] == view(A,Block(1,2))[1,2] == view(A,Block(1,2))[2] == 13

@test view(A, Block(3),Block(1)) ≈ [0,0,0]
@test_throws BandError view(A, Block(3),Block(1))[1,1] = 4
@test_throws BoundsError view(A, Block(5,1))


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

@test BandedMatrix(V) isa BandedMatrix{Int}
@test BandedMatrix{Float64}(V) isa BandedMatrix{Float64}
@test BandedMatrix{Float64}(BandedMatrix(V)) == BandedMatrix{Float64}(V)
@test A[4:6,7:10] ≈ BandedMatrix(V)

@test A[Block(3,4)].l == A.λ
@test A[Block(3,4)].u == A.μ

A[Block(3,4)] = BandedMatrix(Ones{Int}(3,4),(1,1))
@test A[Block(3,4)] == BandedMatrix(Ones{Int}(3,4),(1,1))


if false # turned off since tests have check-bounds=yes
    # test that @inbounds is working properly
    exceed_band(V, k, j) = @inbounds return V[k,j]
    @test exceed_band(V, 2,1) == 8

    @test BandedMatrices.inbands_getindex(V, 2, 1) == 8
    BandedMatrices.inbands_setindex!(V, -2, 5, 1)
    @test A[2,1] == -2
end





l , u = 2,1
λ , μ = 1,2
N = M = 4
cols = rows = 1:N
data = reshape(Vector(1:(λ+μ+1)*(l+u+1)*sum(cols)), (λ+μ+1, (l+u+1)*sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))

# The first block is ignored
@test BlockBandedMatrices.bbb_data_cols(view(A, Block(1,1))) == 2:2
@test BlockBandedMatrices.bbb_data_cols(view(A, Block(2,1))) == 3:3
@test BlockBandedMatrices.bbb_data_cols(view(A, Block(3,1))) == 4:4
@test BlockBandedMatrices.bbb_data_cols(view(A, Block(1,2))) == 5:6
@test BlockBandedMatrices.bbb_data_cols(view(A, Block(2,2))) == 7:8




#### Test Blas arithmetic

l , u = 1,1
λ , μ = 1,1
N = M = 10
cols = rows = fill(1000,N)
data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), (λ+μ+1, (l+u+1)*sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))
V = view(A, Block(N,N))

AN = A[Block(N,N)]
@time BLAS.axpy!(2.0, V, V)
@test A[Block(N,N)] ≈ 3AN


Y = zeros(cols[N], cols[N])
@time BLAS.axpy!(2.0, V, Y)
@test Y ≈ 2A[Block(N,N)]

Y = BandedMatrix(Zeros(cols[N], cols[N]), (λ, μ))
@time BLAS.axpy!(2.0, V, Y)
@test Y ≈ 2A[Block(N,N)]

Y = BandedMatrix(Zeros(cols[N], cols[N]), (λ+1, μ+1))
@time BLAS.axpy!(2.0, V, Y)
@test Y ≈ 2A[Block(N,N)]

Y = BandedMatrix(Zeros(cols[N], cols[N]), (0, 0))
@test_throws BandError BLAS.axpy!(2.0, V, Y)




## standard indexing
l , u = 1,1
λ , μ = 1,1
N = M = 10
cols = rows = 1:N
data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), (λ+μ+1, (l+u+1)*sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))

A[1,1] = 5
@test A[1,1] == 5

@test_throws BandError A[1,4] = 5
A[1,4] = 0
@test A[1,4] == 0

# TODO: return a BandedMatrix
@test A[1:10,1:10] ≈ full(A)[1:10,1:10]


@time A*A
