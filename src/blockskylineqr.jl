qrf!(A,τ) = _qrf!(MemoryLayout(typeof(A)),MemoryLayout(typeof(τ)),A,τ)
_qrf!(::AbstractColumnMajor,::AbstractStridedLayout,A::AbstractMatrix{T},τ::AbstractVector{T}) where T<:BlasFloat =
    LAPACK.geqrf!(A,τ)
_apply_qr!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasReal =
    LAPACK.ormqr!('L','T',A,τ,B)
_apply_qr!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasComplex =
    LAPACK.ormqr!('L','C',A,τ,B)
apply_qr!(A, τ, B) = _apply_qr!(MemoryLayout(typeof(A)), MemoryLayout(typeof(τ)), MemoryLayout(typeof(B)), A, τ, B)

qlf!(A,τ) = _qlf!(MemoryLayout(typeof(A)),MemoryLayout(typeof(τ)),A,τ)
_qlf!(::AbstractColumnMajor,::AbstractStridedLayout,A::AbstractMatrix{T},τ::AbstractVector{T}) where T<:BlasFloat =
    LAPACK.geqlf!(A,τ)
_apply_ql!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasReal =
    LAPACK.ormql!('L','T',A,τ,B)
_apply_ql!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasComplex =
    LAPACK.ormql!('L','C',A,τ,B)
apply_ql!(A, τ, B) = _apply_ql!(MemoryLayout(typeof(A)), MemoryLayout(typeof(τ)), MemoryLayout(typeof(B)), A, τ, B)

function qr!(A::BlockBandedMatrix{T}) where T
    l,u = blockbandwidths(A)
    M,N = blocksize(A)
    ax1 = M < N ? axes(A,1) : axes(A,2)
    τ = PseudoBlockVector{T}(undef, (ax1,))
    for K = 1:min(N,M)
        KR = Block.(K:min(K+l,M))
        V = view(A,KR,Block(K))
        t = view(τ,Block(K))
        qrf!(V,t)
        for J = K+1:min(K+u,N)
            apply_qr!(V, t, view(A,KR,Block(J)))
        end
    end
    QR(A,τ.blocks)
end

function ql!(A::BlockBandedMatrix{T}) where T
    l,u = blockbandwidths(A)
    M,N = blocksize(A)

    ax2 = if M < N
        throw(ArgumentError("Wide block-QL not implented"))
    else
        axes(A,2)
    end
    τ = PseudoBlockVector{T}(undef, (ax2,))
    
    for K = N:-1:max(N - M + 1,1)
        μ = M+K-N
        KR = Block.(max(K-u,1):μ)
        V = view(A,KR,Block(K))
        t = view(τ,Block(K-N+min(M,N)))
        qlf!(V,t)
        for J = K-1:-1:max(K-l,1)
            apply_ql!(V, t, view(A,KR,Block(J)))
        end
    end
    QL(A,τ.blocks)
end

qr(A::BlockBandedMatrix) = qr!(BlockBandedMatrix(A, (blockbandwidth(A,1), blockbandwidth(A,1)+blockbandwidth(A,2))))
ql(A::BlockBandedMatrix) = ql!(BlockBandedMatrix(A, (blockbandwidth(A,1)+blockbandwidth(A,2),blockbandwidth(A,2))))

qr(A::BandedBlockBandedMatrix) = qr(BlockBandedMatrix(A))
ql(A::BandedBlockBandedMatrix) = ql(BlockBandedMatrix(A))

function lmul!(adjQ::Adjoint{<:Any,<:QRPackedQ{<:Any,<:BlockSkylineMatrix}}, Bin::AbstractVector)
    Q = parent(adjQ)
    A = Q.factors
    l,u = blockbandwidths(A)
    N,M = blocksize(A)
    # impose block structure
    ax1 = (axes(A,1),)
    τ  = PseudoBlockArray(Q.τ, ax1)
    B = PseudoBlockArray(Bin, ax1)
    for K = 1:min(N,M)
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
    N,M = blocksize(A)
    # impose block structure
    ax1 = (axes(A,1),)
    τ  = PseudoBlockArray(Q.τ, ax1)
    B = PseudoBlockArray(Bin, ax1)
    for K = N:-1:1
        KR = Block.(max(1,K-u):K)
        V = view(A,KR,Block(K))
        t = view(τ,Block(K))
        apply_ql!(V, t, view(B,KR))
    end
    Bin
end

# avoid LinearALgebra Strided obsession 

for Typ in (:StridedVector, :StridedMatrix, :AbstractVector, :AbstractMatrix)
    @eval function ldiv!(A::QR{<:Any,<:BlockSkylineMatrix}, B::$Typ)
        lmul!(adjoint(A.Q), B)
        M,N = blocksize(A.factors)
        MN = min(M,N)
        V = view(A.factors,Block.(1:MN), Block.(1:MN))
        materialize!(Ldiv(UpperTriangular(V), view(B,1:size(V,1),:)))
        B
    end
end



function ldiv!(A::QL{<:Any,<:BlockSkylineMatrix}, B::AbstractVector)
    lmul!(adjoint(A.Q), B)
    materialize!(Ldiv(LowerTriangular(A.factors), B))
end

function ldiv!(A::QL{<:Any,<:BlockSkylineMatrix}, B::AbstractMatrix)
    lmul!(adjoint(A.Q), B)
    materialize!(Ldiv(LowerTriangular(A.factors), B))
end

\(A::AbstractBlockBandedMatrix, b::AbstractVecOrMat) = qr(A)\b  # use QR because LU would be a _mess_ to implement
