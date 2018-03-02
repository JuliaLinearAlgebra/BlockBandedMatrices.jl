

function _mul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, α, β,
                    ylayout, ::AbstractBlockBandedLayout, xlayout)
    if length(x) != size(A,2) || length(y) != size(A,1)
        throw(BoundsError())
    end

    scale!(β, y)
    o = one(eltype(y))

    for J = Block.(1:nblocks(A,2))
        for K = blockcolrange(A,J)
            kr,jr = globalrange(A.block_sizes, (Int(K),Int(J)))
            mul!(view(y,kr), view(A,K,J), view(x,jr), α, o)
        end
    end
    y
end


function _mul!(Y::AbstractMatrix, A::AbstractMatrix, X::AbstractMatrix, α, β,
                    ::AbstractBlockBandedLayout, ::AbstractBlockBandedLayout, ::AbstractBlockBandedLayout)
    scale!(β, Y)
    o=one(eltype(Y))
    for J=Block(1):Block(nblocks(X,2)),
            N=blockcolrange(X,J), K=blockcolrange(A,N)
        mul!(view(Y,K,J), view(A,K,N), view(X,N,J), α, o)
    end
    Y
end

A_mul_B!(y::AbstractVector, A::AbstractBlockBandedMatrix, b::AbstractVector) =
    mul!(fill!(y, zero(eltype(y))), A, b, one(eltype(A)), zero(eltype(y)))

A_mul_B!(y::AbstractMatrix, A::AbstractBlockBandedMatrix, b::AbstractMatrix) =
    mul!(fill!(y, zero(eltype(y))), A, b, one(eltype(A)), zero(eltype(y)))


#############
# BLAS overrides
#############


function BLAS.axpy!(a, X::AbstractBlockBandedMatrix, Y::AbstractBlockBandedMatrix)
    size(X) == size(Y) || throw(DimensionMismatch())

    for J=Block(1):Block(nblocks(X,2)), K=blockcolrange(X,J)
        BLAS.axpy!(a, view(X,K,J), view(Y,K,J))
    end
    Y
end

BLAS.gemv!(trans::Char, α::T, A::BlockBandedBlock{T}, X::AbstractVector{T}, β::T, Y::AbstractVector{T}) where T <: BlasFloat =
    gemv!(trans, α, A, X, β, Y)


const BBBOrStridedVecOrMat{T} = Union{BlockBandedBlock{T}, StridedVecOrMat{T}}

BLAS.gemm!(transA::Char, transB::Char, α::T, A::BBBOrStridedVecOrMat{T}, B::BBBOrStridedVecOrMat{T}, β::T, C::BBBOrStridedVecOrMat{T}) where T <: BlasFloat =
    gemm!(transA, transB, α, A, B, β, C)


function checkblocks(A, B)
    Arows, Acols = A.block_sizes.block_sizes.cumul_sizes
    Brows, Bcols = B.block_sizes.block_sizes.cumul_sizes
    if Acols ≠ Bcols || Arows ≠ Brows
        throw(DimensionMismatch("*"))
    end
end
## algebra
function fill!(B::BandedBlockBandedBlock{T}, x) where T
    x == zero(T) || throw(BandError(B))
    inblockbands(B) || return B

    fill!(dataview(B), x)
    B
end

function fill!(B::AbstractBlockBandedMatrix{T}, x) where T
    x == zero(T) || throw(BandError(B))

    M,N = nblocks(B)
    for J = 1:N, K = blockcolrange(B,J)
        fill!(view(B,K,Block(J)), x)
    end
    B
end

function blockbanded_copy!(dest::AbstractMatrix{T}, src::AbstractMatrix) where T
    @boundscheck checkblocks(dest, src)

    dl, du = blockbandwidths(dest)
    sl, su = blockbandwidths(src)
    (dl ≥ sl && du ≥ su) || throw(BandError(dest))

    M,N = nblocks(src)
    for J = 1:N
        for K = max(1,J-du):min(J-su-1,M)
            fill!(view(dest,Block(K),Block(J)), zero(T))
        end
        for K = max(1,J-su):min(J+sl,M)
            copy!(view(dest,Block(K),Block(J)), view(src,Block(K),Block(J)))
        end
        for K = max(1,J+sl+1):min(J+dl,M)
            fill!(view(dest,Block(K),Block(J)), zero(T))
        end
    end
    dest
end

copy!(dest::AbstractBlockBandedMatrix, src::AbstractBlockBandedMatrix) =
    blockbanded_copy!(dest, src)

function +(A::BandedBlockBandedMatrix{T}, B::BandedBlockBandedMatrix{V}) where {T<:Number,V<:Number}
    checkblocks(A, B)
    n,m = size(A)
    Arows, Acols = A.block_sizes.block_sizes.cumul_sizes


    bs = BandedBlockBandedSizes(BlockSizes((Arows,Acols)), max(A.l,B.l), max(A.u,B.u), max(A.λ,B.λ), max(A.μ,B.μ))
    TV = promote_type(T,V)
    ret = BandedBlockBandedMatrix{TV}(uninitialized, bs)
    copy!(ret, A)
    BLAS.axpy!(one(TV), B, ret)
end



function *(A::BlockBandedMatrix{T}, B::BlockBandedMatrix{V}) where {T<:Number,V<:Number}
    Arows, Acols = A.block_sizes.block_sizes.cumul_sizes
    Brows, Bcols = B.block_sizes.block_sizes.cumul_sizes
    if Acols ≠ Brows
        # diagonal matrices can be converted
        if isdiag(B) && size(A,2) == size(B,1) == size(B,2)
            B = BlockBandedMatrix(B.data, BlockSizes((Acols,Acols)), 0, 0, 0, 0)
        elseif isdiag(A) && size(A,2) == size(B,1) == size(A,1)
            A = BlockBandedMatrix(A.data, BlockSizes((Brows,Brows)), 0, 0, 0, 0)
        else
            throw(DimensionMismatch("*"))
        end
    end
    n,m = size(A,1), size(B,2)

    l, u = A.l+B.l, A.u+B.u
    A_mul_B!(BlockBandedMatrix{promote_type(T,V)}(uninitialized,
                                    BlockBandedSizes(BlockSizes((Arows,Bcols)), l, u),
                                     l, u),
             A, B)
end



function *(A::BandedBlockBandedMatrix{T}, B::BandedBlockBandedMatrix{V}) where {T<:Number,V<:Number}
    Arows, Acols = A.block_sizes.block_sizes.cumul_sizes
    Brows, Bcols = B.block_sizes.block_sizes.cumul_sizes
    if Acols ≠ Brows
        # diagonal matrices can be converted
        if isdiag(B) && size(A,2) == size(B,1) == size(B,2)
            # TODO: fix
            B = BandedBlockBandedMatrix(B.data, BlockSizes((Acols,Acols)), 0, 0, 0, 0)
        elseif isdiag(A) && size(A,2) == size(B,1) == size(A,1)
            A = BandedBlockBandedMatrix(A.data, BlockSizes((Brows,Brows)), 0, 0, 0, 0)
        else
            throw(DimensionMismatch("*"))
        end
    end
    n,m = size(A,1), size(B,2)

    bs = BandedBlockBandedSizes(BlockSizes((Arows,Bcols)), A.l+B.l, A.u+B.u, A.λ+B.λ, A.μ+B.μ)

    A_mul_B!(BandedBlockBandedMatrix{promote_type(T,V)}(uninitialized, bs),
             A, B)
end




# BlockBandedMatrix with block range indexes is also block-banded
const BlockRange1 = BlockRange{1,Tuple{UnitRange{Int}}}
const BlockBandedSubBlockBandedMatrix{T} =
    SubArray{T,2,BlockBandedMatrix{T},NTuple{2,BlockSlice{BlockRange1}}}


block_sizes(A::AbstractBlockArray) = A.block_sizes
block_sizes(A::AbstractBlockBandedMatrix) = A.block_sizes.block_sizes

function block_sizes(V::BlockBandedSubBlockBandedMatrix)
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindexes(V)[1].block.indices[1]
    JR = parentindexes(V)[2].block.indices[1]

    Bs.cumul_sizes[1]
    @assert KR[1] == JR[1] == 1
    BlockSizes((Bs.cumul_sizes[1][KR[1]:KR[end]+1],Bs.cumul_sizes[1][JR[1]:JR[end]+1]))
end

function blockbandwidths(V::BlockBandedSubBlockBandedMatrix)
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindexes(V)[1].block.indices[1]
    JR = parentindexes(V)[2].block.indices[1]

    @assert KR[1] == JR[1] == 1
    blockbandwidths(A)
end

function blockbandwidth(V::BlockBandedSubBlockBandedMatrix, k::Integer)
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindexes(V)[1].block.indices[1]
    JR = parentindexes(V)[2].block.indices[1]

    @assert KR[1] == JR[1] == 1
    blockbandwidth(A,k)
end

######
# back substitution
######

@inline A_ldiv_B!(U::UpperTriangular{T, BlockBandedBlock{T}}, b::StridedVecOrMat{T}) where {T<:BlasFloat} =
    trtrs!('U', 'N', 'N', parent(U), b)


@inline hasmatchingblocks(A) =
    block_sizes(A).cumul_sizes[1] == block_sizes(A).cumul_sizes[2]

function A_ldiv_B!(U::UpperTriangular{T, BM}, b::StridedVector) where BM <: Union{BlockBandedMatrix{T}, BlockBandedSubBlockBandedMatrix{T}} where T
    A = parent(U)

    @boundscheck size(A,1) == length(b) || throw(BoundsError(A))

    # When blocks are square, use LAPACK trtrs!
    if hasmatchingblocks(A)
        blockbanded_squareblocks_trtrs!(A, b)
    else
        blockbanded_rectblocks_trtrs!(A, b)
    end
end





function blockbanded_squareblocks_trtrs!(A::AbstractMatrix, b::AbstractVector)
    @boundscheck size(A,1) == size(b,1) || throw(BoundsError())

    n = size(b,1)

    Bs = block_sizes(A)
    N = nblocks(Bs,1)

    for K = N:-1:1
        kr = globalrange(Bs, (K,K))[1]
        v = view(b, kr)
        for J = min(N,Int(blockrowstop(A,K))):-1:K+1
            jr = globalrange(Bs, (K,J))[2]
            gemv!('N', -one(eltype(A)), view(A,Block(K),Block(J)), view(b, jr), one(eltype(A)), v)
        end
        @inbounds A_ldiv_B!(UpperTriangular(view(A,Block(K),Block(K))), v)
    end

    b
end


####

const BlockIndexRange1 = BlockIndexRange{1,Tuple{UnitRange{Int64}}}
const DenseBlockSubBlockBandedMatrix{T} = SubArray{T,2,BlockBandedMatrix{T},Tuple{BlockSlice{Block{1,Int}},BlockSlice{BlockIndexRange1}}}
const DenseBlockRangeSubBlockBandedMatrix{T} = SubArray{T,2,BlockBandedMatrix{T},Tuple{BlockSlice{BlockRange1},BlockSlice{BlockIndexRange1}}}

function unsafe_convert(::Type{Ptr{T}}, V::DenseBlockRangeSubBlockBandedMatrix{T}) where T
    A = parent(V)
    JR = parentindexes(V)[2]
    K = first(parentindexes(V)[1].block)
    J = Block(JR)
    K ∈ blockcolrange(A, J) || throw(ArgumentError("Pointer is only defined when inside colrange"))
    p = unsafe_convert(Ptr{T}, view(A, K, J))
    p + sizeof(T)*(JR.block.indices[1][1]-1)*stride(V,2)
end

strides(V::DenseBlockRangeSubBlockBandedMatrix) = (1,parent(V).block_sizes.block_strides[Int(Block(parentindexes(V)[2]))])

struct ShiftedLayout{T,ML<:MemoryLayout} <: MemoryLayout{T}
    shift::Int  # gives the shift to the start of the memory. So shift == 0 implies that the first rows are all zeor
    layout::ML
end

MemoryLayout(V::DenseBlockRangeSubBlockBandedMatrix{T}) where T = ColumnMajor{T}()


# function blockbanded_rectblocks_trtrs!(R::BlockBandedMatrix{T},b::Vector) where T
#     n=n_end=length(b)
#     K_diag=N=Block(R.rowblocks[n])
#     J_diag=M=Block(R.colblocks[n])
#
#     while n > 0
#         B_diag = view(R,K_diag,J_diag)
#
#         kr = blockrows(R,K_diag)
#         jr = blockcols(R,J_diag)
#
#
#         k = n-kr[1]+1
#         j = n-jr[1]+1
#
#         skr = max(1,k-j+1):k   # range in the sub block
#         sjr = max(1,j-k+1):j   # range in the sub block
#
#         kr2 = kr[skr]  # diagonal rows/cols we are working with
#
#         for J = min(M,blockrowstop(R,K_diag)):-1:J_diag+1
#             B=view(R,K_diag,J)
#             Sjr = blockcols(R,J)
#
#             if J==M
#                 Sjr = Sjr[1]:n_end  # The sub rows of the rhs we will multiply
#                 gemv!('N',-one(T),view(B,skr,1:length(Sjr)),
#                                     view(b,Sjr),one(T),view(b,kr2))
#             else  # can use all columns
#                 gemv!('N',-one(T),view(B,skr,:),
#                                     view(b,Sjr),one(T),view(b,kr2))
#             end
#         end
#
#         if J_diag ≠ M && sjr[end] ≠ size(B_diag,2)
#             # subtract non-triangular columns
#             sjr2 = sjr[end]+1:size(B_diag,2)
#             gemv!('N',-one(T),view(B_diag,skr,sjr2),
#                             view(b,sjr2 + jr[1]-1),one(T),view(b,kr2))
#         elseif J_diag == M && sjr[end] ≠ size(B_diag,2)
#             # subtract non-triangular columns
#             Sjr = jr[1]+sjr[end]:n_end
#             gemv!('N',-one(T),view(B_diag,skr,sjr[end]+1:sjr[end]+length(Sjr)),
#                             view(b,Sjr),one(T),view(b,kr2))
#         end
#
#         trtrs!('U','N','N',view(B_diag,skr,sjr),view(b,kr2))
#
#         if k == j
#             K_diag -= 1
#             J_diag -= 1
#         elseif j < k
#             J_diag -= 1
#         else # if k < j
#             K_diag -= 1
#         end
#
#         n = kr2[1]-1
#     end
#     b
# end
#
#
# function trtrs!(A::BlockBandedMatrix{T},u::Matrix) where T
#     if size(A,1) < size(u,1)
#         throw(BoundsError())
#     end
#     n=size(u,1)
#     N=Block(A.rowblocks[n])
#
#     kr1=blockrows(A,N)
#     b=n-kr1[1]+1
#     kr1=kr1[1]:n
#
#     trtrs!('U','N','N',view(A,N[1:b],N[1:b]),view(u,kr1,:))
#
#     for K=N-1:-1:Block(1)
#         kr=blockrows(A,K)
#         for J=min(N,blockrowstop(A,K)):-1:K+1
#             if J==N  # need to take into account zeros
#                 gemm!('N',-one(T),view(A,K,N[1:b]),view(u,kr1,:),one(T),view(u,kr,:))
#             else
#                 gemm!('N',-one(T),view(A,K,J),view(u,blockcols(A,J),:),one(T),view(u,kr,:))
#             end
#         end
#         trtrs!('U','N','N',view(A,K,K),view(u,kr,:))
#     end
#
#     u
# end
