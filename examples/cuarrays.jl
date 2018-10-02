using BandedMatrices, BlockBandedMatrices, LazyArrays, BlockArrays, FillArrays, CuArrays, GPUArrays
import BlockBandedMatrices: _BandedBlockBandedMatrix


function finitedifferences!(data)
    N = nblocks(data,2)
    for J = 1:N
        data.blocks[1,J][1,:] .= 0
        data.blocks[1,J][2,:] .= N^2
        data.blocks[1,J][3,:] .= 0

        data.blocks[2,J][1,:] .= N^2
        data.blocks[2,J][2,:] .= -4N^2
        data.blocks[2,J][3,:] .= N^2

        data.blocks[3,J][1,:] .= 0
        data.blocks[3,J][2,:] .= N^2
        data.blocks[3,J][3,:] .= 0
    end
end


function finitedifferences!(data::AbstractMatrix{T}) where T
    N = nblocks(data,2)
    sc = N^2 * one(T)
    for J = 1:N
        data.blocks[1,J][1,:] .= zero(T)
        data.blocks[2,J][1,:] .= sc
        data.blocks[3,J][1,:] .= zero(T)
    end
    for J = 1:N
        data.blocks[1,J][2,:] .= sc
        data.blocks[2,J][2,:] .= -4sc
        data.blocks[3,J][2,:] .= sc
    end
    for J = 1:N
        data.blocks[1,J][3,:] .= zero(T)
        data.blocks[2,J][3,:] .= sc
        data.blocks[3,J][3,:] .= zero(T)
    end
end



function jl_finitedifferences(N)
    data = BlockMatrix{Float64,JLArray{Float64,2}}(undef, Fill(3,3), Fill(N,N))
    finitedifferences!(data)
    _BandedBlockBandedMatrix(data, (Fill(N,N), Fill(N,N)), (1,1), (1,1))
end


function cu_finitedifferences(::Type{T}, N) where T
    data = BlockMatrix{T,CuArray{T,2}}(undef, Fill(3,3), Fill(N,N))
    finitedifferences!(data)
    _BandedBlockBandedMatrix(data, (Fill(N,N), Fill(N,N)), (1,1), (1,1))
end


T = Float32
b = BlockVector{T, CuArray{T,1}}(undef, Fill(N,N))
c = BlockVector{T, CuArray{T,1}}(Zeros{T}(N^2), Fill(N,N))


function mul!(c, A::BandedBlockBandedMatrix{<:Any,<:BlockMatrix{<:Any,<:CuArray}}, b)
    fill!(c, 0)

    l, u = blockbandwidths(A)
    λ, μ = subblockbandwidths(A)
    N,M = nblocks(A)

    for J = 1:N, K = max(1,J-l):min(N,J+u)
        B = _BandedMatrix(A.data.blocks[K-J+u+1,J],N,λ,μ)
        c.blocks[K] .= Mul(B, b.blocks[J]) .+ c.blocks[K]
    end
    c
end


function mul!(c, A::BandedBlockBandedMatrix{<:Any,<:BlockMatrix{<:Any,<:CuArray}}, b)
    fill!(c, 0)

    l, u = blockbandwidths(A)
    λ, μ = subblockbandwidths(A)
    N,M = nblocks(A)

    for J = 1:N, K = max(1,J-l):min(N,J+u)
        B = _BandedMatrix(A.data.blocks[K-J+u+1,J],N,λ,μ)
        LazyArrays.blasmul!(c.blocks[K],  B, b.blocks[J], 1f0, 1f0)
    end
    c
end

function mul!(c, A::BandedBlockBandedMatrix{<:Any,<:BlockMatrix{<:Any,<:CuArray}}, x)
    fill!(c, 0)

    l, u = blockbandwidths(A)
    λ, μ = subblockbandwidths(A)
    N,M = nblocks(A)

    @inbounds for b = -u:0
        for J = 1-b:M
            K = J + b
            B = _BandedMatrix(A.data.blocks[b+u+1,J],N,λ,μ)
            LazyArrays.blasmul!(c.blocks[K],  B, x.blocks[J], 1f0, 1f0)
        end
    end

    @inbounds for b = 1:l
        for J = 1:M-b
            K = J + b
            B = _BandedMatrix(A.data.blocks[b+u+1,J],N,λ,μ)
            LazyArrays.blasmul!(c.blocks[K],  B, x.blocks[J], 1f0, 1f0)
        end
    end

    c
end



N = 500;
T = Float32
A = cu_finitedifferences(Float32,N)
GPUArrays.synchronize(A)
b = BlockVector{T, CuArray{T,1}}(undef, Fill(N,N))
c = BlockVector{T, CuArray{T,1}}(undef, Fill(N,N))
b .= randn(T,N^2)



function finitedifference_2d(n)
    h = 1/n
    D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2
    D_xx = BandedBlockBandedMatrix(Kron(D², Eye(n)))
    D_yy = BandedBlockBandedMatrix(Kron(Eye(n), D²))
    D_xx + D_yy
end

A_cpu = finitedifference_2d(N)
b_cpu = randn(N^2);
c_cpu = similar(b_cpu)
@time c_cpu .= Mul(A_cpu, b_cpu)l



## large bandwidths
L = U = 20
    N = 500
    A = BandedBlockBandedMatrix{Float64}(undef, (Fill(N,N), Fill(N,N)), (L, U), (L, U))
    A.data.blocks .= randn.()


    b = randn(N^2);
    c = similar(b)
@time c .= Mul(A,b)


function cu_rand(::Type{T}, N, (L, U)) where T
    data = BlockMatrix{T,CuArray{T,2}}(undef, Fill(L+U+1,L+U+1), Fill(N,N))
    _BandedBlockBandedMatrix(data, (Fill(N,N), Fill(N,N)), (L,U), (L,U))
end

T = Float32;
N = 500; L = U = 20; A = cu_rand(T, N, (L,U)); b = BlockVector{T, CuArray{T,1}}(undef, Fill(N,N));
    c = BlockVector{T, CuArray{T,1}}(undef, Fill(N,N));


A_s = sparse(A)

@time A_s*b;
