qrf!(A,τ) = _qrf!(MemoryLayout(A),MemoryLayout(τ),A,τ)
_qrf!(::AbstractColumnMajor,::AbstractStridedLayout,A::AbstractMatrix{T},τ::AbstractVector{T}) where T<:BlasFloat =
    LAPACK.geqrf!(A,τ)
_apply_qr!(::AbstractColumnMajor, ::AbstractStridedLayout, ::AbstractStridedLayout, A::AbstractMatrix{T}, τ::AbstractVector{T}, B::AbstractVecOrMat{T}) where T<:BlasReal =
    LAPACK.ormqr!('L','T',A,τ,B)
apply_qr!(A, τ, B) = _apply_qr!(MemoryLayout(A), MemoryLayout(τ), MemoryLayout(B), A, τ, B)

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

qr(A::BlockBandedMatrix) = qr!(BlockBandedMatrix(A, (blockbandwidth(A,1), blockbandwidth(A,1)+blockbandwidth(A,2))))
