

function _mul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, α, β,
                    ylayout, ::AbstractBlockBandedLayout, xlayout)
    if length(x) != size(A,2) || length(y) != size(A,1)
        throw(BoundsError())
    end

    lmul!(β, y)
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
    lmul!(β, Y)
    o=one(eltype(Y))
    for J=Block(1):Block(nblocks(X,2)),
            N=blockcolrange(X,J), K=blockcolrange(A,N)
        mul!(view(Y,K,J), view(A,K,N), view(X,N,J), α, o)
    end
    Y
end

mul!(y::AbstractVector, A::AbstractBlockBandedMatrix, b::AbstractVector) =
    mul!(fill!(y, zero(eltype(y))), A, b, one(eltype(A)), zero(eltype(y)))

mul!(y::AbstractMatrix, A::AbstractBlockBandedMatrix, b::AbstractMatrix) =
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



const BBBOrStridedVecOrMat{T} = Union{BlockBandedBlock{T}, StridedVecOrMat{T}}

*(V::BlockBandedBlock{T}, b::AbstractVector{T}) where T<:BlasFloat =
    mul!(Array{T}(undef, size(V,1)), V, b, one(T), zero(T))

if VERSION < v"0.7-"
    BLAS.gemv!(trans::Char, α::T, A::BlockBandedBlock{T}, X::AbstractVector{T}, β::T, Y::AbstractVector{T}) where T <: BlasFloat =
        gemv!(trans, α, A, X, β, Y)
    BLAS.gemm!(transA::Char, transB::Char, α::T, A::BBBOrStridedVecOrMat{T}, B::BBBOrStridedVecOrMat{T}, β::T, C::BBBOrStridedVecOrMat{T}) where T <: BlasFloat =
        gemm!(transA, transB, α, A, B, β, C)
end


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

function blockbanded_copyto!(dest::AbstractMatrix{T}, src::AbstractMatrix) where T
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
            copyto!(view(dest,Block(K),Block(J)), view(src,Block(K),Block(J)))
        end
        for K = max(1,J+sl+1):min(J+dl,M)
            fill!(view(dest,Block(K),Block(J)), zero(T))
        end
    end
    dest
end

copyto!(dest::AbstractBlockBandedMatrix, src::AbstractBlockBandedMatrix) =
    blockbanded_copyto!(dest, src)

function +(A::BandedBlockBandedMatrix{T}, B::BandedBlockBandedMatrix{V}) where {T<:Number,V<:Number}
    checkblocks(A, B)
    n,m = size(A)
    Arows, Acols = A.block_sizes.block_sizes.cumul_sizes


    bs = BandedBlockBandedSizes(BlockSizes((Arows,Acols)), max(A.l,B.l), max(A.u,B.u), max(A.λ,B.λ), max(A.μ,B.μ))
    TV = promote_type(T,V)
    ret = BandedBlockBandedMatrix{TV}(undef, bs)
    copyto!(ret, A)
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
    mul!(BlockBandedMatrix{promote_type(T,V)}(undef,
                                    BlockBandedSizes(BlockSizes((Arows,Bcols)), l, u)),
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

    mul!(BandedBlockBandedMatrix{promote_type(T,V)}(undef, bs),
             A, B)
end




# BlockBandedMatrix with block range indexes is also block-banded
const BlockRange1 = BlockRange{1,Tuple{UnitRange{Int}}}
const BlockBandedSubBlockBandedMatrix{T} =
    SubArray{T,2,BlockBandedMatrix{T},NTuple{2,BlockSlice{BlockRange1}}}


block_sizes(A::AbstractBlockArray) = A.block_sizes
block_sizes(A::AbstractBlockBandedMatrix) = A.block_sizes.block_sizes
nblocks(A::AbstractArray) = nblocks(block_sizes(A)) #TODO: Type piracy
nblocks(A::AbstractArray, i::Int) = nblocks(block_sizes(A),i)

function block_sizes(V::BlockBandedSubBlockBandedMatrix)
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindices(V)[1].block.indices[1]
    JR = parentindices(V)[2].block.indices[1]

    Bs.cumul_sizes[1]
    @assert KR[1] == JR[1] == 1
    BlockSizes((Bs.cumul_sizes[1][KR[1]:KR[end]+1],Bs.cumul_sizes[2][JR[1]:JR[end]+1]))
end

function blockbandwidths(V::BlockBandedSubBlockBandedMatrix)
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindices(V)[1].block.indices[1]
    JR = parentindices(V)[2].block.indices[1]

    @assert KR[1] == JR[1] == 1
    blockbandwidths(A)
end

function blockbandwidth(V::BlockBandedSubBlockBandedMatrix, k::Integer)
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindices(V)[1].block.indices[1]
    JR = parentindices(V)[2].block.indices[1]

    @assert KR[1] == JR[1] == 1
    blockbandwidth(A,k)
end



####
# BlockIndexRange subblocks
####

mul!(c::AbstractVector{T}, V::BlockBandedBlock{T}, b::AbstractVector{T}) where T<:BlasFloat =
    BandedMatrices.mul!(c, V, b, one(T), zero(T))

const BlockIndexRange1 = BlockIndexRange{1,Tuple{UnitRange{Int64}}}
const BlockRangeBlockSubBlockBandedMatrix{T} = SubArray{T,2,BlockBandedMatrix{T},Tuple{BlockSlice{BlockRange1},BlockSlice{Block{1,Int}}}}
const BlockRangeBlockIndexRangeSubBlockBandedMatrix{T} = SubArray{T,2,BlockBandedMatrix{T},Tuple{BlockSlice{BlockRange1},BlockSlice{BlockIndexRange1}}}
const BlockIndexRangeBlockIndexRangeSubBlockBandedMatrix{T} = SubArray{T,2,BlockBandedMatrix{T},Tuple{BlockSlice{BlockIndexRange1},BlockSlice{BlockIndexRange1}}}
const BlockBandedSubBlock{T} = BlockIndexRangeBlockIndexRangeSubBlockBandedMatrix{T}


function MemoryLayout(V::BlockRangeBlockSubBlockBandedMatrix)
    A = parent(V)
    KR = parentindices(V)[1].block
    J = parentindices(V)[2].block
    first(KR) ∈ blockcolrange(A, J) || throw(ArgumentError("TODO: Use ShiftedLayout"))
    ColumnMajor()
end

strides(V::BlockRangeBlockSubBlockBandedMatrix) =
    (1,parent(V).block_sizes.block_strides[Int(parentindices(V)[2].block)])

function unsafe_convert(::Type{Ptr{T}}, V::BlockRangeBlockSubBlockBandedMatrix{T}) where T
    A = parent(V)
    JR = parentindices(V)[2]
    KR = parentindices(V)[1].block
    J = parentindices(V)[2].block
    p = unsafe_convert(Ptr{T}, view(A, first(KR), J))
end

*(V::BlockRangeBlockSubBlockBandedMatrix{T}, b::AbstractVector{T}) where T<:BlasFloat =
    mul!(Array{T}(undef, size(V,1)), V, b, one(T), zero(T))


if VERSION < v"0.7-"
    BLAS.gemv!(trans::Char, α::T, A::BlockRangeBlockSubBlockBandedMatrix{T}, X::AbstractVector{T}, β::T, Y::AbstractVector{T}) where T <: BlasFloat =
        gemv!(trans, α, A, X, β, Y)
end

# struct ShiftedLayout{T,ML<:MemoryLayout} <: MemoryLayout
#     shift::Tuple{Int,Int}  # gives the shift to the start of the memory.
#                            # So shift == (0,0) is equivalent to layout
#                            # shift == (2,1) has the first two rows and first column all zero
#     layout::ML
# end

function MemoryLayout(V::BlockRangeBlockIndexRangeSubBlockBandedMatrix)
    A = parent(V)
    JR = parentindices(V)[2]
    K = first(parentindices(V)[1].block)
    J = Block(JR)
    K ∈ blockcolrange(A, J) || throw(ArgumentError("TODO: Use ShiftedLayout"))
    ColumnMajor()
end

strides(V::BlockRangeBlockIndexRangeSubBlockBandedMatrix) =
    (1,parent(V).block_sizes.block_strides[Int(Block(parentindices(V)[2]))])

function unsafe_convert(::Type{Ptr{T}}, V::BlockRangeBlockIndexRangeSubBlockBandedMatrix{T}) where T
    A = parent(V)
    JR = parentindices(V)[2]
    K = first(parentindices(V)[1].block)
    J = Block(JR)
    K ∈ blockcolrange(A, J) || throw(ArgumentError("Pointer is only defined when inside colrange"))
    p = unsafe_convert(Ptr{T}, view(A, K, J))
    p + sizeof(T)*(JR.block.indices[1][1]-1)*stride(V,2)
end

*(V::BlockRangeBlockIndexRangeSubBlockBandedMatrix{T}, b::AbstractVector{T}) where T<:BlasFloat =
    mul!(Array{T}(undef, size(V,1)), V, b, one(T), zero(T))

if VERSION < v"0.7-"
    BLAS.gemv!(trans::Char, α::T, A::BlockRangeBlockIndexRangeSubBlockBandedMatrix{T}, X::AbstractVector{T}, β::T, Y::AbstractVector{T}) where T <: BlasFloat =
        gemv!(trans, α, A, X, β, Y)
end

function unsafe_convert(::Type{Ptr{T}}, V::BlockBandedSubBlock{T}) where T
    A = parent(V)
    JR = parentindices(V)[2]
    K = parentindices(V)[1].block.block
    kr = parentindices(V)[1].block.indices[1]
    J = parentindices(V)[2].block.block
    jr = parentindices(V)[2].block.indices[1]
    p = unsafe_convert(Ptr{T}, view(A, K, J))
    p + sizeof(T)*(kr[1]-1 + (jr[1]-1)*stride(V,2))
end

strides(V::BlockBandedSubBlock) =
    (1,parent(V).block_sizes.block_strides[Int(parentindices(V)[2].block.block)])

MemoryLayout(V::BlockBandedSubBlock) = ColumnMajor()
*(V::BlockBandedSubBlock{T}, b::AbstractVector{T}) where T<:BlasFloat =
    mul!(Array{T}(undef, size(V,1)), V, b, one(T), zero(T))

if VERSION < v"0.7-"
    BLAS.gemv!(trans::Char, α::T, A::BlockBandedSubBlock{T}, X::AbstractVector{T}, β::T, Y::AbstractVector{T}) where T <: BlasFloat =
        gemv!(trans, α, A, X, β, Y)
end

######
# back substitution
######

@inline ldiv!(U::UpperTriangular{T, BLOCK}, b::StridedVecOrMat{T}) where BLOCK <: Union{BlockBandedBlock{T}, BlockBandedSubBlock{T}} where T<:BlasFloat =
    trtrs!('U', 'N', 'N', parent(U), b)

@inline ldiv!(U::UpperTriangular{T, BlockBandedBlock{T}}, b::StridedVecOrMat{T}) where {T<:BlasFloat} =
    trtrs!('U', 'N', 'N', parent(U), b)


@inline hasmatchingblocks(A) =
    block_sizes(A).cumul_sizes[1] == block_sizes(A).cumul_sizes[2]

function ldiv!(U::UpperTriangular{T, BM}, b::StridedVector) where BM <: Union{BlockBandedMatrix{T}, BlockBandedSubBlockBandedMatrix{T}} where T
    A = parent(U)

    @boundscheck size(A,1) == length(b) || throw(BoundsError(A))

    # When blocks are square, use LAPACK trtrs!
    if hasmatchingblocks(A)
        blockbanded_squareblocks_trtrs!(A, b)
    else
        blockbanded_rectblocks_trtrs!(A, b)
    end
end

function blockbanded_squareblocks_trtrs!(A::AbstractMatrix{T}, b::AbstractVector{T}) where T
    @boundscheck size(A,1) == size(b,1) || throw(BoundsError())

    n = size(b,1)

    Bs = block_sizes(A)
    N = nblocks(Bs,1)

    for K = N:-1:1
        V_22 = view(A, Block(K),  Block(K))
        b_2 = view(b, parentindices(V_22)[1].indices)
        ldiv!(UpperTriangular(V_22), b_2)

        V_12 = view(A, blockcolstart(A, K):Block(K-1), Block(K))
        b̃_1 = view(b, parentindices(V_12)[1].indices)
        mul!(b̃_1, V_12, b_2, -one(T), one(T))
    end

    b
end

# we want to make sure the block are matching up to the blocksize
function hasmatchingblocks(V::SubArray{T,2,BlockBandedMatrix{T},Tuple{UnitRange{Int},UnitRange{Int}}}) where T
    A = parent(V)
    kr, jr = parentindices(V)
    N,  N_n = _find_block(block_sizes(A), 1, kr[end])
    M,  M_n = _find_block(block_sizes(A), 2, jr[end])
    N == M && hasmatchingblocks(view(A, Block.(1:N), Block.(1:N)))
end

# Write U as [U_11 U_12; 0 U_22] and b = [b_1,b_2,b_3] to use efficient block versions
function ldiv!(U::UpperTriangular{T,SV},
                   b::AbstractVector{T}) where SV<:SubArray{T,2,BlockBandedMatrix{T},Tuple{UnitRange{Int},UnitRange{Int}}} where T
    V = parent(U)
    if hasmatchingblocks(V)
        blockbanded_squareblocks_intrange_trtrs!(V, b)
    else
        blockbanded_rectblocks_intrange_trtrs!(V, b)
    end
end

function blockbanded_squareblocks_intrange_trtrs!(V::AbstractMatrix{T}, b::AbstractVector{T}) where T
    A = parent(V)
    kr, jr = parentindices(V)

    N,  N_n = _find_block(block_sizes(A), 1, kr[end])

    V_22 = view(A, Block(N)[1:N_n],  Block(N)[1:N_n])
    b_2 = view(b, parentindices(V_22)[1].indices)
    ldiv!(UpperTriangular(V_22), b_2)

    V_12 = view(A, blockcolstart(A, N):Block(N-1),  Block(N)[1:N_n])
    b̃_1 = view(b, parentindices(V_12)[1].indices)
    mul!(b̃_1, V_12, b_2, -one(T), one(T))

    V_11 = view(A, Block.(1:N-1), Block.(1:N-1))
    b_1 = view(b, parentindices(V_11)[1].indices)
    ldiv!(UpperTriangular(V_11), b_1)
    b
end


function squaredblocks(bs::BlockSizes{2})
    new_blocks = sort(union(bs.cumul_sizes...))
    BlockSizes((new_blocks,new_blocks))
end

function _squaredblocks_newbandwidth(l, kr, jr, cs)
    l_ret = 0
    j_old = 2
    for j = 2:length(cs)
        j_old = 2
        while cs[j] > jr[j_old]
            j_old += 1
        end
        k_old = kr[min(j_old+l,length(kr))]
        if k_old ≥ last(cs)
            l_ret = max(l_ret,last(cs)-j)
        else
            l_ret = max(l_ret,findfirst(cs, k_old)-j)
        end
    end
    l_ret
end

function squaredblocks(bs::BlockBandedSizes)
    l, u = blockbandwidths(bs)


    kr, jr = bs.block_sizes.cumul_sizes
    kr[end] == jr[end] || throw(ArgumentError("Can only turn a square matrix into squared blocks"))


    new_bs = squaredblocks(bs.block_sizes)
    cs = new_bs.cumul_sizes[1]

    new_l, new_u = _squaredblocks_newbandwidth(l, kr, jr, cs), _squaredblocks_newbandwidth(u, jr, kr, cs)
    BlockBandedSizes(new_bs, new_l, new_u)
end

function _squaredblocks_mapback(kr, cs)
    N = length(cs)-1
    ret = Vector{BlockIndexRange1}(undef, N)
    K_old = 1
    for K = 1:N
        cs[K] ≥ kr[K_old+1] && (K_old += 1)
        ret[K] = Block(K_old)[cs[K]-kr[K_old]+1:cs[K+1]-kr[K_old]]
    end
    ret
end


block_banded_sizes(A::BlockBandedMatrix) = A.block_sizes
block_banded_sizes(A::AbstractMatrix) = BlockBandedSizes(block_sizes(A), blockbandwidths(A)...)


function blockbanded_rectblocks_trtrs!(A::AbstractMatrix{T}, b::AbstractVector{T}) where T
    @boundscheck size(A,1) == size(b,1) || throw(BoundsError())
    bbs = block_banded_sizes(A)
    bs_square = squaredblocks(bbs)
    l, u = blockbandwidths(A)
    l_new, u_new = blockbandwidths(bs_square)
    cs = bs_square.block_sizes.cumul_sizes[1]

    KR, JR = bbs.block_sizes.cumul_sizes

    KR_map = _squaredblocks_mapback(KR, cs)
    JR_map = _squaredblocks_mapback(JR, cs)

    N = length(KR_map)

    for J = N:-1:1
        V_22 = view(A, KR_map[J], JR_map[J])
        b_2  = view(b, parentindices(V_22)[1].indices)
        ldiv!(UpperTriangular(V_22), b_2)

        for K = max(1,J-u_new):J-1
            if KR_map[K].block ≥ JR_map[J].block - u # inside old blockbandwith
                V_12 = view(A, KR_map[K], JR_map[J])
                kr_sub = parentindices(V_12)[1].indices
                b̃_1 = view(b, kr_sub)
                mul!(b̃_1, V_12, b_2, -one(T), one(T))
            end
        end
    end

    b
end

# Make KR stop at size n
function _cumul_maxsize!(KR, n)
    for k in eachindex(KR)
        if KR[k] ≥ n+1
            resize!(KR, k-1)
            push!(KR, n+1)
            break
        end
    end
    KR
end

function squaredblocks(bs::BlockBandedSizes, n::Int)
    l, u = blockbandwidths(bs)
    kr, jr = bs.block_sizes.cumul_sizes
    new_bs = squaredblocks(bs.block_sizes)
    cs = new_bs.cumul_sizes[1]
    _cumul_maxsize!(cs,n)
    new_l, new_u = _squaredblocks_newbandwidth(l, kr, jr, cs), _squaredblocks_newbandwidth(u, jr, kr, cs)
    BlockBandedSizes(new_bs, new_l, new_u)
end



function blockbanded_rectblocks_intrange_trtrs!(V::AbstractMatrix{T}, b::AbstractVector{T}) where T
    @assert 1 == first(parentindices(V)[1]) == first(parentindices(V)[2])
    P = parent(V)
    n, m = size(V)
    N, N_n = _find_block(block_sizes(P), 1, n)
    M, M_n = _find_block(block_sizes(P), 2, m)

    A = view(P, Block.(1:N), Block.(1:M))
    bbs = block_banded_sizes(A)
    KR, JR = bbs.block_sizes.cumul_sizes
    bs_square = squaredblocks(bbs, n)
    l_new, u_new = blockbandwidths(bs_square)

    cs = bs_square.block_sizes.cumul_sizes[1]
    KR_map = _squaredblocks_mapback(KR, cs)
    JR_map = _squaredblocks_mapback(JR, cs)

    l, u = blockbandwidths(bbs)

    l_new, u_new = blockbandwidths(bs_square)

    N = length(KR_map)

    for J = N:-1:1
        V_22 = view(A, KR_map[J], JR_map[J])
        b_2  = view(b, parentindices(V_22)[1].indices)
        ldiv!(UpperTriangular(V_22), b_2)

        for K = max(1,J-u_new):J-1
            if KR_map[K].block ≥ JR_map[J].block - u # inside old blockbandwith
                V_12 = view(A, KR_map[K], JR_map[J])
                kr_sub = parentindices(V_12)[1].indices
                b̃_1 = view(b, kr_sub)
                mul!(b̃_1, V_12, b_2, -one(T), one(T))
            end
        end
    end

    b
end



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
