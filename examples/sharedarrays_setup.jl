using Pkg

Pkg.activate(homedir() * "/Documents/Coding/gpublockbanded")
using BandedMatrices, BlockBandedMatrices, SharedArrays, LazyArrays, BlockArrays, FillArrays
import BlockBandedMatrices: _BandedBlockBandedMatrix, BandedBlockBandedSizes, BlockSizes, blockcolrange
import BandedMatrices: AbstractBandedMatrix, bandwidths, BandedStyle
import Base: getindex, size

struct FiniteDifference{T} <: AbstractBandedMatrix{T}
    n::Int
end

FiniteDifference(n) = FiniteDifference{Float64}(n)

getindex(F::FiniteDifference{T}, k::Int, j::Int) where T =
    if k == j
        -2*one(T)*F.n^2
    elseif abs(k-j) == 1
        one(T)*F.n^2
    else
        zero(T)
    end

bandwidths(F::FiniteDifference) = (1,1)
size(F::FiniteDifference) = (F.n,F.n)

Base.BroadcastStyle(::Type{<:Eye}) = BandedStyle()


function blockcolbuild!(ret, A, JR)
    for J in JR, K in blockcolrange(A,J)
        view(ret,K,J) .= view(A,K,J)
    end
end
