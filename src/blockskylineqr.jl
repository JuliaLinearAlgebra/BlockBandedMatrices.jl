_apply_qr!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasReal =
    LAPACK.ormqr!('L','T',A,τ,B)
_apply_qr!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasComplex =
    LAPACK.ormqr!('L','C',A,τ,B)
_apply_qr!(_, _, _, A::AbstractMatrix, τ::AbstractVector, B::AbstractVecOrMat) = lmul!(MatrixFactorizations.QRPackedQ(A, τ)', B)
apply_qr!(A, τ, B) = _apply_qr!(MemoryLayout(A), MemoryLayout(τ), MemoryLayout(B), A, τ, B)

qlf!(A,τ) = _qlf!(MemoryLayout(A),MemoryLayout(τ),A,τ)
_qlf!(::AbstractColumnMajor,::AbstractStridedLayout,A::AbstractMatrix{T},τ::AbstractVector{T}) where T<:BlasFloat =
    LAPACK.geqlf!(A,τ)
_apply_ql!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasReal =
    LAPACK.ormql!('L','T',A,τ,B)
_apply_ql!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasComplex =
    LAPACK.ormql!('L','C',A,τ,B)
apply_ql!(A, τ, B) = _apply_ql!(MemoryLayout(A), MemoryLayout(τ), MemoryLayout(B), A, τ, B)

function _blockbanded_qr!(A::AbstractMatrix, τ::AbstractVector)
    M,N = Block.(blocksize(A))
    (M < N ? axes(A,1) : axes(A,2)) == axes(τ,1) || throw(DimensionMismatch(""))
    _blockbanded_qr!(A, τ, min(N,M))
    QR(A,τ.blocks)
end

function _blockbanded_qr!(A::AbstractMatrix, τ::AbstractVector, NCOLS::Block{1})
    l,u = blockbandwidths(A)
    M,N = Block.(blocksize(A))
    for K = Block(1):NCOLS
        KR = K:min(K+l,M)
        V = view(A,KR,K)
        t = view(τ,K)
        MatrixFactorizations.qrfactUnblocked!(V,t)
        for J = K+1:min(K+u,N)
            apply_qr!(V, t, view(A,KR,J))
        end
    end
    A,τ
end

function qr!(A::BlockBandedMatrix{T}) where T
    M,N = blocksize(A)
    ax1 = M < N ? axes(A,1) : axes(A,2)
    _blockbanded_qr!(A, BlockedVector(zeros(T,length(ax1)), (ax1,)))
end

function ql!(A::BlockBandedMatrix{T}) where T
    l,u = blockbandwidths(A)
    M,N = blocksize(A)

    ax2 = if M < N
        throw(ArgumentError("Wide block-QL not implented"))
    else
        axes(A,2)
    end
    τ = BlockedVector{T}(undef, (ax2,))

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

_qr(::AbstractBlockBandedLayout, ::Tuple{Integer,Integer}, A) = qr!(BlockBandedMatrix(A, (blockbandwidth(A,1), blockbandwidth(A,1)+blockbandwidth(A,2))))
_qr(lay::AbstractBlockBandedLayout, ax::Tuple{AbstractUnitRange{Int},AbstractUnitRange{Int}}, A) = _qr(lay, map(length, ax), A)
_ql(::AbstractBlockBandedLayout, _, A) = ql!(BlockBandedMatrix(A, (blockbandwidth(A,1)+blockbandwidth(A,2),blockbandwidth(A,2))))
_factorize(::AbstractBlockBandedLayout, _, A) = qr(A)

function materialize!(Mul::MatLmulVec{<:AdjQRPackedQLayout{<:AbstractBlockBandedLayout}})
    adjQ,Bin = Mul.A,Mul.B
    Q = parent(adjQ)
    A = Q.factors
    l,u = blockbandwidths(A)
    N,M = blocksize(A)
    # impose block structure
    ax1,ax2 = axes(A)
    τ  = BlockedArray(Q.τ, (length(ax1) ≤ length(ax2) ? ax1 : ax2,))
    B = BlockedArray(Bin, (ax1,))
    for K = 1:min(N,M)
        KR = Block.(K:min(K+l,N))
        V = view(A,KR,Block(K))
        t = view(τ,Block(K))
        apply_qr!(V, t, view(B,KR))
    end
    Bin
end
function materialize!(Mul::MatLmulVec{<:AdjQLPackedQLayout{<:AbstractBlockBandedLayout}})
    Q = parent(Mul.A)
    A = Q.factors
    l,u = blockbandwidths(A)
    N,M = blocksize(A)
    # impose block structure
    ax1,ax2 = axes(A)
    τ  = BlockedArray(Q.τ, (length(ax1) ≤ length(ax2) ? ax1 : ax2,))
    B = BlockedArray(Mul.B, (ax1,))
    for K = N:-1:1
        KR = Block.(max(1,K-u):K)
        V = view(A,KR,Block(K))
        t = view(τ,Block(K))
        apply_ql!(V, t, view(B,KR))
    end
    Mul.B
end

function materialize!(Mul::MatLmulMat{<:AdjQRPackedQLayout{<:AbstractBlockBandedLayout}})
    adjQ,Bin = Mul.A,Mul.B
    Q = parent(adjQ)
    A = Q.factors
    l,u = blockbandwidths(A)
    N,M = blocksize(A)
    # impose block structure
    ax1,ax2 = axes(A)
    τ  = BlockedArray(Q.τ, (length(ax1) ≤ length(ax2) ? ax1 : ax2,))
    B = BlockedArray(Bin, (ax1,axes(Bin,2)))
    for K = 1:min(N,M), J = 1:blocksize(Bin,2)
        KR = Block.(K:min(K+l,N))
        V = view(A,KR,Block(K))
        t = view(τ,Block(K))
        apply_qr!(V, t, view(B,KR,Block(J)))
    end
    Bin
end

# avoid LinearALgebra Strided obsession

for Typ in (:StridedVector, :StridedMatrix, :AbstractVector, :AbstractMatrix, :LayoutMatrix, :LayoutVector)
    @eval function ldiv!(A::QR{<:Any,<:BlockSkylineMatrix}, B::$Typ)
        lmul!(adjoint(A.Q), B)
        M,N = blocksize(A.factors)
        if M == N
            materialize!(Ldiv(UpperTriangular(A.factors), view(B,1:size(A.factors,1),:)))
        else
            MN = min(M,N)
            V = view(A.factors,Block.(1:MN), Block.(1:MN))
            materialize!(Ldiv(UpperTriangular(V), view(B,1:size(V,1),:)))
        end
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
