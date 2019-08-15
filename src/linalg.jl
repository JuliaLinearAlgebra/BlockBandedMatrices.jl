# BlockBandedMatrix with block range indexes is also block-banded
const Block1 = Block{1,Int}
const BlockRange1 = BlockRange{1,Tuple{UnitRange{Int}}}
const BlockIndexRange1 = BlockIndexRange{1,Tuple{UnitRange{Int}}}
const SubBlockSkylineMatrix{T,LL,UU,R1,R2} =
    SubArray{T,2,BlockSkylineMatrix{T,LL,UU},Tuple{BlockSlice{R1},BlockSlice{R2}}}

const SubBandedBlockBandedMatrix{T,R1,R2} =
    SubArray{T,2,<:BandedBlockBandedMatrix{T},Tuple{BlockSlice{R1},BlockSlice{R2}}}



getindex(A::BandedBlockBandedMatrix, KR::BlockRange1, JR::BlockRange1) = BandedBlockBandedMatrix(view(A, KR, JR))
getindex(A::BandedBlockBandedMatrix, KR::BlockRange1, J::Block1) = BandedBlockBandedMatrix(view(A, KR, J))
getindex(A::BandedBlockBandedMatrix, K::Block1, JR::BlockRange1) = BandedBlockBandedMatrix(view(A, K, JR))


@lazymul AbstractBlockBandedMatrix


MemoryLayout(A::Type{<:PseudoBlockArray{T,N,R}}) where {T,N,R} = MemoryLayout(R)


#############
# BLAS overrides
#############

function materialize!(M::MatMulVecAdd{<:AbstractBlockBandedLayout,<:AbstractStridedLayout,<:AbstractStridedLayout})
    α, A, x_in, β, y_in = M.α, M.A, M.B, M.β, M.C
    if length(x_in) != size(A,2) || length(y_in) != size(A,1)
        throw(DimensionMismatch())
    end

    # impose block structure
    y = PseudoBlockArray(y_in, BlockSizes((cumulsizes(blocksizes(A),1),)))
    x = PseudoBlockArray(x_in, BlockSizes((cumulsizes(blocksizes(A),2),)))

    _fill_lmul!(β, y)

    for J = Block.(1:nblocks(A,2))
        for K = blockcolrange(A,J)
            view(y,K) .= α .* Mul(view(A,K,J), view(x,J)) .+ view(y,K)
        end
    end
    y_in
end

function materialize!(M::MatMulMatAdd{<:AbstractBlockBandedLayout,<:AbstractBlockBandedLayout,<:AbstractBlockBandedLayout})
    α, A, X, β, Y = M.α, M.A, M.B, M.β, M.C
    _fill_lmul!(β, Y)
    for J=Block(1):Block(nblocks(X,2)),
            N=blockcolrange(X,J), K=blockcolrange(A,N)
        view(Y,K,J) .= α .* Mul( view(A,K,N), view(X,N,J)) .+ view(Y,K,J)
    end
    Y
end

function materialize!(M::MatMulMatAdd{<:AbstractBlockBandedLayout,<:AbstractColumnMajor,<:AbstractColumnMajor})
    α, A, X_in, β, Y_in = M.α, M.A, M.B, M.β, M.C
    _fill_lmul!(β, Y_in)
    X = PseudoBlockArray(X_in, BlockSizes((cumulsizes(blocksizes(A),2),[1,size(X_in,2)+1])))
    Y = PseudoBlockArray(Y_in, BlockSizes((cumulsizes(blocksizes(A),1), [1,size(Y_in,2)+1])))
    for N=Block.(1:nblocks(X,1)), K=blockcolrange(A,N)
        view(Y,K,Block(1)) .= α .* Mul( view(A,K,N), view(X,N,Block(1))) .+ view(Y,K,Block(1))
    end
    Y_in
end

function materialize!(M::MatMulMatAdd{<:AbstractColumnMajor,<:AbstractBlockBandedLayout,<:AbstractColumnMajor})
    α, A_in, X, β, Y_in = M.α, M.A, M.B, M.β, M.C
    _fill_lmul!(β, Y_in)
    A = PseudoBlockArray(A_in, BlockSizes(([1,size(A_in,1)+1],cumulsizes(blocksizes(X),1))))
    Y = PseudoBlockArray(Y_in, BlockSizes(([1,size(Y_in,1)+1],cumulsizes(blocksizes(X),2))))
    for J=Block(1):Block(nblocks(X,2)), N=blockcolrange(X,J)
        view(Y,Block(1),J) .= α .* Mul( view(A,Block(1),N), view(X,N,J)) .+ view(Y,Block(1),J)
    end
    Y_in
end




#############
# * overrides
#############

*(A::BlockBandedMatrix, B::BlockBandedMatrix) = materialize(Mul(A,B))
*(A::BlockBandedMatrix, B::BandedBlockBandedMatrix) = materialize(Mul(A,B))
*(A::BandedBlockBandedMatrix, B::BlockBandedMatrix) = materialize(Mul(A,B))
*(A::BandedBlockBandedMatrix, B::BandedBlockBandedMatrix) = materialize(Mul(A,B))
*(A::Matrix, B::BlockBandedMatrix) = materialize(Mul(A,B))
*(A::BlockBandedMatrix, B::Matrix) = materialize(Mul(A,B))
*(A::BandedBlockBandedMatrix, B::Matrix) = materialize(Mul(A,B))
*(A::Matrix, B::BandedBlockBandedMatrix) = materialize(Mul(A,B))


function add_bandwidths(A::AbstractBlockBandedMatrix,B::AbstractBlockBandedMatrix)
    Al,Au = colblockbandwidths(A)
    Bl,Bu = colblockbandwidths(B)

    l = Vector(Bl)
    u = Vector(Bu)

    for (v,Av) in [(l,Al),(u,Au)]
        n = length(v)
        for i = 1:n
            sel = max(i-Bu[i],1):min(i+Bl[i],length(Av))
            isempty(sel) && continue
            v[i] += maximum(Av[sel])
        end
    end

    l,u
end

function add_bandwidths(A::BlockBandedMatrix,B::BlockBandedMatrix)
    l,u = blockbandwidths(A) .+ blockbandwidths(B)
    Fill(l,nblocks(B,2)), Fill(u,nblocks(B,2))
end

struct BlockBandedMulAddStyle <: AbstractMulAddStyle end
struct BandedBlockBandedMulAddStyle <: AbstractMulAddStyle end

mulapplystyle(A::AbstractBlockBandedLayout, B::AbstractBlockBandedLayout) = BlockBandedMulAddStyle()
mulapplystyle(A::AbstractBlockBandedLayout, B::DiagonalLayout) = mulapplystyle(A,A)
mulapplystyle(A::DiagonalLayout, B::AbstractBlockBandedLayout) = mulapplystyle(B,B)
mulapplystyle(A::BandedBlockBandedColumnMajor, B::BandedBlockBandedColumnMajor) = BandedBlockBandedMulAddStyle()

function similar(M::Mul{BlockBandedMulAddStyle}, ::Type{T}) where T
    A,B = M.args
    A isa Diagonal && return similar(B,T)
    B isa Diagonal && return similar(A,T)

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

    l,u = add_bandwidths(A,B)
    BlockSkylineMatrix{T}(undef, BlockSkylineSizes(BlockSizes((Arows,Bcols)), l, u))
end

function similar(M::Mul{BandedBlockBandedMulAddStyle}, ::Type{T}) where T
    A,B = M.args
    A isa Diagonal && return similar(B,T)
    B isa Diagonal && return similar(A,T)

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

    BandedBlockBandedMatrix{T}(undef, bs)
end




function blocksizes(V::SubBlockSkylineMatrix{<:Any,LL,UU,BlockRange1,BlockRange1}) where {LL,UU}
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindices(V)[1].block.indices[1]
    JR = parentindices(V)[2].block.indices[1]
    shift = Int(KR[1])-Int(JR[1])

    Bs.cumul_sizes[1]
    @assert KR[1] == JR[1] == 1
    BlockSkylineSizes(BlockSizes((Bs.cumul_sizes[1][KR[1]:KR[end]+1] .- Bs.cumul_sizes[1][KR[1]] .+ 1,
                                       Bs.cumul_sizes[2][JR[1]:JR[end]+1] .- Bs.cumul_sizes[2][JR[1]] .+ 1)),
                           colblockbandwidth(A,1)[1:Int(JR[end])] .- shift, colblockbandwidth(A,2)[1:Int(JR[end])] .+ shift)
end

function blockbandwidths(V::SubBlockSkylineMatrix{<:Any,LL,UU,BlockRange1,BlockRange1}) where {LL,UU}
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindices(V)[1].block.indices[1]
    JR = parentindices(V)[2].block.indices[1]

    @assert KR[1] == JR[1] == 1
    blockbandwidths(A)
end


####
# BlockIndexRange subblocks
####


subarraylayout(::BlockBandedColumnMajor, ::Type{Tuple{BlockSlice{BlockRange1}, BlockSlice{BlockRange1}}}) = BlockBandedColumnMajor()
subarraylayout(::BlockBandedColumnMajor, ::Type{Tuple{BlockSlice{BlockRange1}, BlockSlice{Block1}}}) = ColumnMajor()
subarraylayout(::BlockBandedColumnMajor, ::Type{Tuple{BlockSlice{BlockRange1}, BlockSlice{BlockIndexRange1}}}) = ColumnMajor()
subarraylayout(::BlockBandedColumnMajor, ::Type{Tuple{BlockSlice{BlockIndexRange1}, BlockSlice{BlockIndexRange1}}}) = ColumnMajor()

subarraylayout(::BandedBlockBandedColumnMajor, ::Type{Tuple{BlockSlice{Block1}, BlockSlice{Block1}}}) = BandedColumnMajor()
subarraylayout(::BandedBlockBandedColumnMajor, ::Type{Tuple{BlockSlice{BlockRange1}, BlockSlice{BlockRange1}}}) = BandedBlockBandedColumnMajor()
subarraylayout(::BandedBlockBandedColumnMajor, ::Type{Tuple{BlockSlice{Block1}, BlockSlice{BlockRange1}}}) = BandedBlockBandedColumnMajor()
subarraylayout(::BandedBlockBandedColumnMajor, ::Type{Tuple{BlockSlice{BlockRange1}, BlockSlice{Block1}}}) = BandedBlockBandedColumnMajor()
subarraylayout(::BandedBlockBandedColumnMajor, ::Type{Tuple{BlockSlice{BlockRange1}, BlockSlice{BlockIndexRange1}}}) = BandedBlockBandedColumnMajor()
subarraylayout(::BandedBlockBandedColumnMajor, ::Type{Tuple{BlockSlice{BlockIndexRange1}, BlockSlice{BlockIndexRange1}}}) = BandedBlockBandedColumnMajor()

isbanded(A::SubArray{<:Any,2,<:BandedBlockBandedMatrix}) = MemoryLayout(typeof(A)) == BandedColumnMajor()
isbandedblockbanded(A::SubArray{<:Any,2,<:BandedBlockBandedMatrix}) = MemoryLayout(typeof(A)) == BandedBlockBandedColumnMajor()


subblockbandwidths(V::SubBandedBlockBandedMatrix) = subblockbandwidths(parent(V))

function blockbandwidths(V::SubBandedBlockBandedMatrix{<:Any,BlockRange1,Block1})
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindices(V)[1].block.indices[1]
    J = parentindices(V)[2].block
    shift = Int(KR[1])-Int(J)
    blockbandwidth(A,1) - shift, blockbandwidth(A,2) + shift
end

function blockbandwidths(V::SubBandedBlockBandedMatrix{<:Any,Block1,BlockRange1})
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    K = parentindices(V)[1].block
    JR = parentindices(V)[2].block.indices[1]
    shift = Int(K)-Int(JR[1])

    blockbandwidth(A,1) - shift, blockbandwidth(A,2) + shift
end

function blockbandwidths(V::SubBandedBlockBandedMatrix{<:Any,BlockRange1,BlockRange1})
    A = parent(V)
    Bs = A.block_sizes.block_sizes

    KR = parentindices(V)[1].block.indices[1]
    JR = parentindices(V)[2].block.indices[1]
    shift = Int(KR[1])-Int(JR[1])

    blockbandwidth(A,1) - shift, blockbandwidth(A,2) + shift
end


strides(V::SubBlockSkylineMatrix{<:Any,LL,UU,<:Union{BlockRange1,Block1},Block1}) where {LL,UU} =
    (1,parent(V).block_sizes.block_strides[Int(parentindices(V)[2].block)])


function unsafe_convert(::Type{Ptr{T}}, V::SubBlockSkylineMatrix{T,LL,UU,<:Union{BlockRange1,Block1},Block1}) where {T,LL,UU}
    A = parent(V)
    JR = parentindices(V)[2]
    KR = parentindices(V)[1].block
    J = parentindices(V)[2].block
    p = unsafe_convert(Ptr{T}, view(A, first(KR), J))
end

struct ShiftedLayout{T,ML<:MemoryLayout} <: MemoryLayout
    shift::Tuple{Int,Int}  # gives the shift to the start of the memory.
                           # So shift == (0,0) is equivalent to layout
                           # shift == (2,1) has the first two rows and first column all zero
    layout::ML
end


strides(V::SubBlockSkylineMatrix{<:Any,LL,UU,BlockRange1,BlockIndexRange1}) where {LL,UU} =
    (1,parent(V).block_sizes.block_strides[Int(Block(parentindices(V)[2]))])

function unsafe_convert(::Type{Ptr{T}}, V::SubBlockSkylineMatrix{T,LL,UU,BlockRange1,BlockIndexRange1}) where {T,LL,UU}
    A = parent(V)
    JR = parentindices(V)[2]
    K = first(parentindices(V)[1].block)
    J = Block(JR)
    K ∈ blockcolrange(A, J) || throw(ArgumentError("Pointer is only defined when inside colrange"))
    p = unsafe_convert(Ptr{T}, view(A, K, J))
    p + sizeof(T)*(JR.block.indices[1][1]-1)*stride(V,2)
end

function unsafe_convert(::Type{Ptr{T}}, V::SubBlockSkylineMatrix{T,LL,UU,BlockIndexRange1,BlockIndexRange1}) where {T,LL,UU}
    A = parent(V)
    JR = parentindices(V)[2]
    K = parentindices(V)[1].block.block
    kr = parentindices(V)[1].block.indices[1]
    J = parentindices(V)[2].block.block
    jr = parentindices(V)[2].block.indices[1]
    p = unsafe_convert(Ptr{T}, view(A, K, J))
    p + sizeof(T)*(kr[1]-1 + (jr[1]-1)*stride(V,2))
end

strides(V::SubBlockSkylineMatrix{T,LL,UU,BlockIndexRange1,BlockIndexRange1}) where {T,LL,UU} =
    (1,parent(V).block_sizes.block_strides[Int(parentindices(V)[2].block.block)])

MemoryLayout(V::SubBlockSkylineMatrix{T,LL,UU,BlockIndexRange1,BlockIndexRange1}) where {T,LL,UU} = ColumnMajor()


