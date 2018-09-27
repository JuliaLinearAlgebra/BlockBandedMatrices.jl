using Distributed

pids = addprocs(4)
@everywhere using Pkg
@everywhere Pkg.activate(homedir() * "/Documents/Coding/gpublockbanded")
@everywhere using BandedMatrices, BlockBandedMatrices, SharedArrays, LazyArrays, BlockArrays, FillArrays
@everywhere import BlockBandedMatrices: _BandedBlockBandedMatrix, BandedBlockBandedSizes, BlockSizes, blockcolrange



function shared_BandedBlockBandedMatrix(::UndefInitializer, bs::BlockSizes, (l, u), (λ, μ); pids=Int[])
    bs = BandedBlockBandedSizes(bs, l, u, λ, μ)
    data = SharedMatrix{Float64}((l+u+1)*(λ+μ+1), size(A,2); pids=pids)
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


N = 600
    h = 1/N
    D² = BandedMatrix(0 => Fill(-2,N), 1 => Fill(1,N-1), -1 => Fill(1,N-1))/h^2
    A = Kron(D², Eye(N))
    @time ret = shared_BandedBlockBandedMatrix(A; pids=pids)
    @time ret = BandedBlockBandedMatrix(A)
