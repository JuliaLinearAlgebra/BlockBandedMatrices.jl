using BandedMatrices, BlockBandedMatrices, CuArrays, LazyArrays, BlockArrays, FillArrays
import BlockBandedMatrices: _BandedBlockBandedMatrix, BandedBlockBandedSizes, BlockSizes, blockcolrange
import BandedMatrices: AbstractBandedMatrix, bandwidths
import Base: getindex, size

function cu_BandedBlockBandedMatrix(::UndefInitializer, bs::BlockSizes, (l, u), (λ, μ))
    bs = BandedBlockBandedSizes(bs, l, u, λ, μ)
    data = CuArray{Float64}((l+u+1)*(λ+μ+1), size(A,2))
    _BandedBlockBandedMatrix(PseudoBlockArray(data, bs.data_block_sizes), bs)
end

shared_BandedBlockBandedMatrix(::UndefInitializer, (N,M), (l, u), (λ, μ); pids=Int[]) =
    shared_BandedBlockBandedMatrix(undef, BlockSizes(N,M), (l, u), (λ, μ); pids=pids)


function shared_BandedBlockBandedMatrix(A::AbstractMatrix; pids=Int[])
    ret = shared_BandedBlockBandedMatrix(undef, blocksizes(A), (1,1), (0,0); pids=pids)
    N,M = nblocks(A)
    @sync @distributed for J = Block.(1:N)
        for K = blockcolrange(A,J)
            view(ret,K,J) .= view(A,K,J)
        end
    end
    ret
end

@everywhere struct FiniteDifference{T} <: AbstractBandedMatrix{T}
    n::Int
end

@everywhere FiniteDifference(n) = FiniteDifference{Float64}(n)

@everywhere getindex(F::FiniteDifference{T}, k::Int, j::Int) where T =
    if k == j
        -2*one(T)*F.n^2
    elseif abs(k-j) == 1
        one(T)*F.n^2
    else
        zero(T)
    end

@everywhere bandwidths(F::FiniteDifference) = (1,1)
@everywhere size(F::FiniteDifference) = (F.n,F.n)



N = 1000
    h = 1/N
    D² = FiniteDifference(N)
    A = Kron(D², Eye(N))
    A
@time ret = shared_BandedBlockBandedMatrix(A; pids=pids)
@time ret = BandedBlockBandedMatrix(A)
