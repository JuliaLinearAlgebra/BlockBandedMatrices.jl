using BlockBandedMatrices

A = BandedBlockBandedMatrix{Float64}(undef, (1:10,1:10), (1,1), (1,1))
    A.data .= randn.()
    A
U = UpperTriangular(A)

using Test
import LazyArrays: MemoryLayout, UpperTriangularLayout
@test MemoryLayout(U) == UpperTriangularLayout(MemoryLayout(A))


view(U, Block(1,1))
