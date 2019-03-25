qrf!(A,τ) = _qrf!(MemoryLayout(A),MemoryLayout(τ),A,τ)
_qrf!(::AbstractColumnMajor,::AbstractStridedLayout,A::AbstractMatrix{T},τ::AbstractVector{T}) where T<:BlasFloat =
    LAPACK.geqrf!(A,τ)
_apply_qr!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasReal =
    LAPACK.ormqr!('L','T',A,τ,B)
apply_qr!(A, τ, B) = _apply_qr!(MemoryLayout(A), MemoryLayout(τ), MemoryLayout(B), A, τ, B)

qlf!(A,τ) = _qlf!(MemoryLayout(A),MemoryLayout(τ),A,τ)
_qlf!(::AbstractColumnMajor,::AbstractStridedLayout,A::AbstractMatrix{T},τ::AbstractVector{T}) where T<:BlasFloat =
    LAPACK.geqlf!(A,τ)
_apply_ql!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasReal =
    LAPACK.ormql!('L','T',A,τ,B)
apply_ql!(A, τ, B) = _apply_ql!(MemoryLayout(A), MemoryLayout(τ), MemoryLayout(B), A, τ, B)

function qr!(A::BlockBandedMatrix)
    bs = BlockSizes((cumulsizes(blocksizes(A),1),))
    τ = PseudoBlockVector{Float64}(undef, bs)
    l,u = blockbandwidths(A)
    N,M = nblocks(A)
    for K = 1:N
        KR = Block.(K:min(K+l,N))
        V = view(A,KR,Block(K))
        t = view(τ,Block(K))
        qrf!(V,t)
        for J = K+1:min(K+u,M)
            apply_qr!(V, t, view(A,KR,Block(J)))
        end
    end
    QR(A,τ.blocks)
end

function ql!(A::BlockBandedMatrix)
    bs = BlockSizes((cumulsizes(blocksizes(A),1),))
    τ = PseudoBlockVector{Float64}(undef, bs)
    l,u = blockbandwidths(A)
    N,M = nblocks(A)
    for K = N:-1:1
        KR = Block.(max(K-u,1):K)
        V = view(A,KR,Block(K))
        t = view(τ,Block(K))
        qlf!(V,t)
        for J = K-1:-1:max(K-l,1)
            apply_ql!(V, t, view(A,KR,Block(J)))
        end
    end
    QL(A,τ.blocks)
end

qr(A::BlockBandedMatrix) = qr!(BlockBandedMatrix(A, (blockbandwidth(A,1), blockbandwidth(A,1)+blockbandwidth(A,2))))
ql(A::BlockBandedMatrix) = ql!(BlockBandedMatrix(A, (blockbandwidth(A,1)+blockbandwidth(A,2),blockbandwidth(A,2))))

function lmul!(adjQ::Adjoint{<:Any,<:QRPackedQ{<:Any,<:BlockSkylineMatrix}}, Bin::AbstractVector)
    Q = parent(adjQ)
    A = Q.factors
    l,u = blockbandwidths(A)
    N,M = nblocks(A)
    # impose block structure
    bs = BlockSizes((cumulsizes(blocksizes(A),1),))
    τ  = PseudoBlockArray(Q.τ, bs)
    B = PseudoBlockArray(Bin, bs)
    for K = 1:N
        KR = Block.(K:min(K+l,N))
        V = view(A,KR,Block(K))
        t = view(τ,Block(K))
        apply_qr!(V, t, view(B,KR))
    end
    Bin
end

function lmul!(adjQ::Adjoint{<:Any,<:QLPackedQ{<:Any,<:BlockSkylineMatrix}}, Bin::AbstractVector)
    Q = parent(adjQ)
    A = Q.factors
    l,u = blockbandwidths(A)
    N,M = nblocks(A)
    # impose block structure
    bs = BlockSizes((cumulsizes(blocksizes(A),1),))
    τ  = PseudoBlockArray(Q.τ, bs)
    B = PseudoBlockArray(Bin, bs)
    for K = N:-1:1
        KR = Block.(max(1,K-u):K)
        V = view(A,KR,Block(K))
        t = view(τ,Block(K))
        apply_ql!(V, t, view(B,KR))
    end
    Bin
end

# avoid LinearALgebra Strided obsession 

function ldiv!(A::QL{<:Any,<:BlockSkylineMatrix}, B::AbstractVector)
    lmul!(adjoint(A.Q), B)
    B .= Ldiv(LowerTriangular(A.factors), B)
end

function ldiv!(A::QL{<:Any,<:BlockSkylineMatrix}, B::AbstractMatrix)
    lmul!(adjoint(A.Q), B)
    B .= Ldiv(LowerTriangular(A.factors), B)
end

\(A::AbstractBlockBandedMatrix, b::AbstractVecOrMat) = qr(A)\b  # use QR because LU would be a _mess_ to implement