using BlockArrays, BandedMatrices, BlockBandedMatrices, Compat.Test
    import BlockBandedMatrices: _BandedBlockBandedMatrix, blockcolrange, blockrowrange, colrange, rowrange

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



data = reshape(collect(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))
@test Matrix(BlockBandedMatrix(A)) == Matrix(A)


@test blockbandwidths(A) == (l,u)
@test BlockBandedMatrices.subblockbandwidths(A) == (l,u)


@test blockrowrange(A, 1) == Block.(1:2)
@test blockrowrange(A, 2) == Block.(1:3)
@test blockrowrange(A, 3) == Block.(2:4)

@test blockcolrange(A, 1) == Block.(1:2)
@test blockcolrange(A, 2) == Block.(1:3)
@test blockcolrange(A, 3) == Block.(2:4)

@test rowrange(A,1) == 1:3
@test rowrange(A,2) == 1:6
@test rowrange(A,3) == 1:6
@test rowrange(A,4) == 2:10


@test colrange(A,1) == 1:3
@test colrange(A,2) == 1:6
@test colrange(A,3) == 1:6
@test colrange(A,4) == 2:10



# check views of blocks are indexing correctly

@test A[Block(1), Block(1)] isa BandedMatrix
@test A[Block(1), Block(1)] == A[Block(1,1)] == BlockArrays.getblock(A, 1, 1) == BandedMatrix(view(A, Block(1,1)))
@test A[1,1] == view(A,Block(1),Block(1))[1,1] == view(A,Block(1,1))[1,1] == A[Block(1,1)][1,1]  == A[Block(1),Block(1)][1,1] == 5
@test A[2,1] == view(A,Block(2),Block(1))[1,1] == view(A,Block(2,1))[1,1] == 8
@test A[3,1] == view(A,Block(2),Block(1))[2,1] == 9
@test A[4,1] == 0
@test A[1,2] == view(A,Block(1,2))[1,1] == 11
@test A[1,3] == view(A,Block(1,2))[1,2] == view(A,Block(1,2))[2] == 19

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

@test A.l == l == blockbandwidth(A,1)
@test A.u == u == blockbandwidth(A,2)
@test blockbandwidths(A) == (l, u)

@test A.λ == λ == subblockbandwidth(A,1)
@test A.μ == μ == subblockbandwidth(A,2)
@test subblockbandwidths(A) == (λ, μ)

@test blockrowrange(A, 1) == Block.(1:2)
@test blockrowrange(A, 2) == Block.(1:3)
@test blockrowrange(A, 3) == Block.(1:4)
@test blockrowrange(A, 4) == Block.(2:4)


@test blockcolrange(A, 1) == Block.(1:3)
@test blockcolrange(A, 2) == Block.(1:4)
@test blockcolrange(A, 3) == Block.(2:4)
@test blockcolrange(A, 4) == Block.(3:4)

@test rowrange(A,1) == 1:3
@test rowrange(A,2) == 1:6
@test rowrange(A,3) == 1:6
@test rowrange(A,4) == 1:10

@test colrange(A,1) == 1:6
@test colrange(A,2) == 1:10
@test colrange(A,3) == 1:10
@test colrange(A,4) == 2:10

@test_throws BandError A[1,4] = 5
@test_throws BandError view(A, Block(1,3))[2] = 5


#### Test Blas arithmetic

l , u = 1,1
λ , μ = 1,1
N = M = 10
cols = rows = fill(1000,N)
data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
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
data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1),sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))

A[1,1] = 5
@test A[1,1] == 5

@test_throws BandError A[1,4] = 5
A[1,4] = 0
@test A[1,4] == 0

# TODO: return a BandedMatrix
@test A[1:10,1:10] ≈ full(A)[1:10,1:10]


@time A*A



lu = (l , u) = -1,1
λμ = (λ , μ) = 0,1
rows = 1:5
cols = 1:6

data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), (λ+μ+1, (l+u+1)*sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))
@test_throws BandError A[1,1] = 5

@test A[1,2] == 4
@test A[Block(2,2)] == [0 0; 0 0]
@test A[Block(2,3)]  == [8.0 9.0 0.0; 0.0 10.0 11.0]
@test bandwidths(A[Block(2,3)]) == (0,1)

@test blockrowrange(A, 1) == Block.(2:2)
@test blockrowrange(A, 2) == Block.(3:3)
@test blockrowrange(A, 3) == Block.(4:4)
@test blockrowrange(A, 4) == Block.(5:5)

@test blockcolrange(A, 1) == Block.(1:0)
@test blockcolrange(A, 2) == Block.(1:1)
@test blockcolrange(A, 3) == Block.(2:2)

@test rowrange(A,1) == 2:3
@test rowrange(A,2) == 4:6
@test rowrange(A,3) == 4:6
@test rowrange(A,4) == 7:10


@test colrange(A,1) == 1:0
@test colrange(A,2) == 1:1
@test colrange(A,3) == 1:1
@test colrange(A,4) == 2:3


lu = (l , u) = -1,1
λμ = (λ , μ) = -1,1
rows = 1:5
cols = 1:6

data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), (λ+μ+1, (l+u+1)*sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))
@test_throws BandError A[1,1] = 5

@test A[1,3] == 3
@test A[Block(2,2)] == [0 0; 0 0]
@test A[Block(2,3)]  == [0 5 0; 0 0 6]
@test bandwidths(A[Block(2,3)]) == (-1,1)



@test blockcolrange(A, 1) == Block.(1:0)
@test blockcolrange(A, 2) == Block.(1:1)
@test blockcolrange(A, 3) == Block.(2:2)

@test colrange(A,1) == 1:0
@test colrange(A,2) == 1:1
@test colrange(A,3) == 1:1
@test colrange(A,4) == 2:3


### test other types

A = BandedBlockBandedMatrix{Float32}(Zeros{Float32}(10,10),
                            (ones(Int,10), ones(Int,10)), (1,1), (1,1))

@test eltype(A) == Float32


A = BandedBlockBandedMatrix(Zeros{Float32}(10,10),
                            (ones(Int,10), ones(Int,10)), (1,1), (1,1))

@test eltype(A) == Float32
