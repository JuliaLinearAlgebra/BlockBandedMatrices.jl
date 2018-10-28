using BandedMatrices, BlockBandedMatrices, LazyArrays, BlockArrays, FillArrays, CuArrays, GPUArrays, LinearAlgebra
import BlockBandedMatrices: _BandedBlockBandedMatrix
import BandedMatrices: _BandedMatrix


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


function finitedifferences(::Type{Mat}, N) where Mat<:AbstractMatrix
    data = BlockMatrix{eltype(Mat),Mat}(undef, Fill(3,3), Fill(N,N))
    finitedifferences!(data)
    _BandedBlockBandedMatrix(data, (Fill(N,N), Fill(N,N)), (1,1), (1,1))
end

finitedifferences(::Type{Arr}, N) where Arr<:AbstractArray = FiniteDifference(Arr{2}, N)

T = Float32
N = 10
x = y = range(zero(T), stop=one(T), length=N)

u₀ = BlockVector{T, Vec}(undef, Fill(N,N))






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
        LazyArrays.materialize!(MulAdd(1f0, B, b.blocks[J], 1f0, c.blocks[K]))
    end
    c
end

function mul2!(c, A::BandedBlockBandedMatrix{<:Any,<:BlockMatrix{<:Any,<:CuArray}}, x)
    fill!(c, 0)

    l, u = blockbandwidths(A)
    λ, μ = subblockbandwidths(A)
    N,M = nblocks(A)

    @inbounds for b = -u:0
        for J = 1-b:M
            K = J + b
            B = _BandedMatrix(A.data.blocks[b+u+1,J],N,λ,μ)
            LazyArrays.materialize!(MulAdd(1f0, B, x.blocks[J], 1f0, c.blocks[K]))
        end
    end

    @inbounds for b = 1:l
        for J = 1:M-b
            K = J + b
            B = _BandedMatrix(A.data.blocks[b+u+1,J],N,λ,μ)
            LazyArrays.materialize!(MulAdd(1f0, B, x.blocks[J], 1f0, c.blocks[K]))
        end
    end

    c
end

function mul3!(c, A::BandedBlockBandedMatrix{T,<:BlockMatrix{<:Any}}, x) where T
    # fill!(c, 0)

    l, u = blockbandwidths(A)
    λ, μ = subblockbandwidths(A)
    N,M = nblocks(A)

    @inbounds for b = -u:0
        for J = 1-b:M
            K = J + b
            BLAS.gbmv!('N', N, λ, μ,  one(T), A.data.blocks[b+u+1,J], x.blocks[J], one(T), c.blocks[K])
        end
    end

    @inbounds for b = 1:l
        for J = 1:M-b
            K = J + b
            BLAS.gbmv!('N', N, λ, μ,  one(T), A.data.blocks[b+u+1,J], x.blocks[J], one(T), c.blocks[K])
        end
    end

    c
end


function mul_diag!(c, A::BandedBlockBandedMatrix{<:Any,<:BlockMatrix{<:Any,<:CuArray}}, x)
    # fill!(c, 0)

    l, u = blockbandwidths(A)
    λ, μ = subblockbandwidths(A)
    N,M = nblocks(A)

    for J = 1:M
        BLAS.gbmv!('N', N, λ, μ,  1f0, A.data.blocks[2,J], x.blocks[J], 1f0, c.blocks[J])
    end
    c
end


N = 4000;
T = Float32
A = finitedifferences(CuArray{T,2},N)
x = BlockVector{T, CuArray{T,1}}(undef, Fill(N,N))
c = BlockVector{T, CuArray{T,1}}(undef, Fill(N,N))


A_cpu = finitedifferences(Array{T,2},N)
x_cpu = BlockVector{T, Array{T,1}}(undef, Fill(N,N))
c_cpu = BlockVector{T, Array{T,1}}(undef, Fill(N,N))

@time mul3!(c, A, x);
@time mul3!(c_cpu, A_cpu, x_cpu);


GPUArrays.synchronize(A)
x = BlockVector{T, CuArray{T,1}}(undef, Fill(N,N))
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
@time c_cpu .= Mul(A_cpu, b_cpu)



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


n = 1_000_000; A = CuArray{Float32}(3,n); B = similar(A); C = similar(A);
    x = CuArray{Float32}(n); y = similar(x); z = similar(x);
    a = similar(x);  b = similar(x); c = similar(x);
    @time multgbmv!((A,B,C), (x,y,z), (a,b,c), n);


function multgbmv!((A,B,C), (x,y,z), (a,b,c), n)
   BLAS.gbmv!('N', n, 1, 1, 1f0, A, x, 0f0, a)
   BLAS.gbmv!('N', n, 1, 1, 1f0, B, y, 0f0, b)
   BLAS.gbmv!('N', n, 1, 1, 1f0, C, z, 0f0, x)
end
