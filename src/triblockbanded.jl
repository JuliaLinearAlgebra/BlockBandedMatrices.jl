block_sizes(A::AbstractTriangular) = block_sizes(parent(A))
#TODO: non-matchin g blocks
isblockbanded(A::AbstractTriangular) =
    isblockbanded(parent(A)) && hasmatchingblocks(A)
isbandedblockbanded(A::AbstractTriangular) =
    isbandedblockbanded(parent(A)) && hasmatchingblocks(A)
blockbandwidths(A::Union{UpperTriangular,UnitUpperTriangular}) = let P = parent(A)
        (min(0,blockbandwidths(P,1)), blockbandwidth(P,2))
    end
bandwidths(A::Union{LowerTriangular,UnitLowerTriangular}) = let P = parent(A)
        (blockbandwidth(P,1), min(0,bandwidth(P,2)))
    end
subblockbandwidths(A::AbstractTriangular) = subblockbandwidths(parent(A))

triangularlayout(::Type{Tri}, ML::BandedBlockBandedColumnMajor) where {Tri} = Tri(ML)


A = BandedBlockBandedMatrix{Float64}(undef, (1:10,1:10), (1,1), (1,1))
    A.data .= randn.()
U = UpperTriangular(A)

using Test
import LazyArrays: MemoryLayout, UpperTriangularLayout
@test MemoryLayout(U) == UpperTriangularLayout(MemoryLayout(A))


view(U, Block(1,1))
