using Distributed

pids = addprocs(4)
@everywhere using Pkg
@everywhere Pkg.activate(homedir() * "/Documents/Coding/gpublockbanded")
@everywhere using BandedMatrices, BlockBandedMatrices, SharedArrays, LazyArrays, BlockArrays, FillArrays
@everywhere import BlockBandedMatrices: _BandedBlockBandedMatrix, BandedBlockBandedSizes, BlockSizes, blockcolrange
@everywhere import BandedMatrices: AbstractBandedMatrix, bandwidths
@everywhere import Base: getindex, size

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


function shared_BandedBlockBandedMatrix(A::AbstractMatrix; pids=Int[])
    ret = shared_BandedBlockBandedMatrix(undef, blocksizes(A), (1,1), (0,0); pids=pids)
    N,M = nblocks(A)
    @sync @distributed for J = Block.(1:N)
            view(ret,J,J) .= view(A,J,J)
    end
    ret
end

@everywhere function diag_build!(ret, A, JR)
    for J in JR
        # @show J
        view(ret,Block(J,J)) .= view(A,Block(J,J))
    end
end



function populate!(ret, A::AbstractMatrix)
    pids = procs(ret.data.blocks)
    N,M = nblocks(A)
    bl = N ÷ length(pids)
    @sync begin
        for k = 1:length(pids)
            JR = ((k-1)*bl+1):(k*bl)
            # @show k, JR
            @async remotecall_wait(diag_build!, pids[k], ret, A, JR)
        end
    end
    ret
end



function populate1!(ret, A::AbstractMatrix)
    pids = procs(ret.data.blocks)
    N,M = nblocks(A)
    bl = N ÷ length(pids)
    @sync begin
        for k = 1:1
            JR = 1:N
            # @show k, JR
            @async remotecall_wait(diag_build!, pids[k], ret, A, JR)
        end
    end
    ret
end




procs(ret.data.blocks)



N = 4 * 200
    h = 1/N
    D² = FiniteDifference(N)
    A = Kron(D², Eye(N))
    A
    pids = [6,7,8,9]
    ret = shared_BandedBlockBandedMatrix(undef, blocksizes(A), (1,1), (0,0); pids=pids)


A

print("Serial: ");  @time BandedBlockBandedMatrix(A)
print("remotecall: "); @time populate!(ret, A)
print("remotecall1: "); @time populate1!(ret, A)


@which BandedBlockBandedMatrix(A)


ret = BandedBlockBandedMatrix{Float64}(undef, blocksizes(A), (1,1), (0,0))

@time diag_build!(ret, A, 1:N)


ret

ret[Block(N,N)]

procs(ret.data.blocks)

ret


pids

pids2

nblocks(A)[1] , length(pids)

using Profile
Profile.clear()
ret.data


ret = shared_BandedBlockBandedMatrix(undef, blocksizes(A), (1,1), (0,0); pids=pids)
N,M = nblocks(A)
@sync @distributed for J = Block.(1:N)
    for K = blockcolrange(A,J)
        view(ret,K,J) .= view(A,K,J)
    end
end
ret
