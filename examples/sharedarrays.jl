using Distributed

pids = addprocs(4)

@everywhere include((@__DIR__)*"/sharedarrays_setup.jl")

function shared_BandedBlockBandedMatrix(::UndefInitializer, bs::BlockSizes, (l, u), (λ, μ); pids=Int[])
    bs = BandedBlockBandedSizes(bs, l, u, λ, μ)
    data = SharedMatrix{Float64}((l+u+1)*(λ+μ+1), size(A,2); pids=pids)
    _BandedBlockBandedMatrix(PseudoBlockArray(data, bs.data_block_sizes), bs)
end

shared_BandedBlockBandedMatrix(::UndefInitializer, (N,M), (l, u), (λ, μ); pids=Int[]) =
    shared_BandedBlockBandedMatrix(undef, BlockSizes(N,M), (l, u), (λ, μ); pids=pids)

function populate!(ret, A::AbstractMatrix)
    pids = procs(ret.data.blocks)
    N,M = nblocks(A)
    bl = N ÷ length(pids)
    @sync begin
        for k = 1:length(pids)
            JR = Block.(((k-1)*bl+1):(k*bl))  #TODO: case where length(pids)  does not divide N
            @async remotecall_wait(blockcolbuild!, pids[k], ret, A, JR)
        end
    end
    ret
end

function shared_BandedBlockBandedMatrix(A::AbstractMatrix; pids=Int[])
    ret = shared_BandedBlockBandedMatrix(undef, blocksizes(A), (1,1), (0,0); pids=pids)
    populate!(ret, A)
end



N = 4 * 1000
    h = 1/N
    D² = FiniteDifference(N)
    A = Kron(D², Eye(N))
@time BandedBlockBandedMatrix(A)           # 5s
@time shared_BandedBlockBandedMatrix(A)    # 0.9s
