using BlockArrays, BandedMatrices, BlockBandedMatrices, Base.Test

l = u = 1
λ = μ = 1
N = M = 10
cols = rows = 1:N
data = ones(λ+μ+1, (l+u+1)*sum(cols))
A = BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))
# test blocks
V = view(A, Block(2,2))
@test V[1,1] ≈ 1
@test_throws BoundsError V[1,4]
V[1,1] = 3
@test V[1,1] ≈ 3
@test A[2,2] ≈ 3

@test_throws BoundsError view(A, Block(2,11))

v=ones(size(A,1))
M=full(A)
@test norm(A*v-M*v) ≤ 100eps()

i,j = 6,7
bi = BlockArrays.global2blockindex(A.block_sizes, (i,j))


A
@inbounds v = view(A, Block(bi.I))[bi.α]

ty

BlockArrays.global2blockindex(A.block_sizes, (5,6))
