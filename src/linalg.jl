# BlockBandedMatrix with block range indexes is also block-banded
const SubBlockSkylineMatrix{T,LL,UU,R1,R2} =
    SubArray{T,2,BlockSkylineMatrix{T,LL,UU},<:Tuple{<:BlockSlice{R1},<:BlockSlice{R2}}}




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
    Fill(l,blocksize(B,2)), Fill(u,blocksize(B,2))
end

function similar(M::MulAdd{<:AbstractBlockBandedLayout,<:AbstractBlockBandedLayout}, ::Type{T}) where T
    A,B = M.A, M.B

    if !blockisequal(axes(A,2), axes(B,1))
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
    BlockSkylineMatrix{T}(undef, (axes(A,1),axes(B,2)), (l,u))
end

function similar(M::MulAdd{BandedBlockBandedColumnMajor,BandedBlockBandedColumnMajor}, ::Type{T}) where T
    A,B = M.A, M.B

    if !blockisequal(axes(A,2), axes(B,1))
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

    BandedBlockBandedMatrix{T}(undef, (axes(A,1),axes(B,2)), (A.l+B.l, A.u+B.u), (A.λ+B.λ, A.μ+B.μ))
end

similar(M::MulAdd{<:DiagonalLayout,<:AbstractBlockBandedLayout}, ::Type{T}) where T = similar(M.B,T)
similar(M::MulAdd{<:AbstractBlockBandedLayout,<:DiagonalLayout}, ::Type{T}) where T = similar(M.A,T)



function blockbandwidths(V::SubBlockSkylineMatrix{<:Any,LL,UU,<:BlockRange1,<:BlockRange1}) where {LL,UU}
    A = parent(V)

    KR = parentindices(V)[1].block.indices[1]
    JR = parentindices(V)[2].block.indices[1]

    shift =  first(KR) - first(JR)
    l,u = blockbandwidths(A)
    l-shift, u+shift
end


####
# BlockIndexRange subblocks
####

sublayout(::AbstractBlockBandedLayout, ::Type{<:Tuple{<:BlockSlice{<:BlockRange1}, <:BlockSlice{<:BlockRange1}}}) = BlockBandedLayout()

sublayout(::BlockBandedColumnMajor, ::Type{<:Tuple{<:BlockSlice{<:BlockRange1}, <:BlockSlice{<:BlockRange1}}}) = BlockBandedColumnMajor()
sublayout(::BlockBandedColumnMajor, ::Type{<:Tuple{<:BlockSlice{<:BlockRange1}, <:BlockSlice{Block1}}}) = ColumnMajor()
sublayout(::BlockBandedColumnMajor, ::Type{<:Tuple{<:BlockSlice{<:BlockRange1}, <:BlockSlice{<:BlockIndexRange1}}}) = ColumnMajor()
sublayout(::BlockBandedColumnMajor, ::Type{<:Tuple{<:BlockSlice{<:BlockIndexRange1}, <:BlockSlice{<:BlockIndexRange1}}}) = ColumnMajor()

isblockbanded(V::SubArray{<:Any,2,<:Any,<:Tuple{<:BlockSlice{<:BlockRange1}, <:BlockSlice{<:BlockRange1}}}) =
    isblockbanded(parent(V))

sub_materialize(::AbstractBlockBandedLayout, V, _) = BlockBandedMatrix(V)
sub_materialize(::AbstractBlockBandedLayout, V, ::Tuple{<:BlockedUnitRange,<:BlockedUnitRange}) = BlockBandedMatrix(V)
sub_materialize(::BlockLayout{<:AbstractBandedLayout}, V, _) = BlockBandedMatrix(V)
sub_materialize(::BlockLayout{<:AbstractBandedLayout}, V, ::Tuple{<:BlockedUnitRange,<:BlockedUnitRange}) = BlockBandedMatrix(V)

strides(V::SubBlockSkylineMatrix{<:Any,LL,UU,<:Union{BlockRange1,Block1},Block1}) where {LL,UU} =
    (1,parent(V).block_sizes.block_strides[Int(parentindices(V)[2].block)])


function unsafe_convert(::Type{Ptr{T}}, V::SubBlockSkylineMatrix{T,LL,UU,<:Union{BlockRange1,Block1},Block1}) where {T,LL,UU}
    A = parent(V)
    JR = parentindices(V)[2]
    KR = parentindices(V)[1].block
    J = parentindices(V)[2].block
    p = unsafe_convert(Ptr{T}, view(A, first(KR), J))
end

strides(V::SubBlockSkylineMatrix{<:Any,LL,UU,<:BlockRange1,<:BlockIndexRange1}) where {LL,UU} =
    (1,parent(V).block_sizes.block_strides[Int(Block(parentindices(V)[2]))])

function unsafe_convert(::Type{Ptr{T}}, V::SubBlockSkylineMatrix{T,LL,UU,<:BlockRange1,<:BlockIndexRange1}) where {T,LL,UU}
    A = parent(V)
    JR = parentindices(V)[2]
    K = first(parentindices(V)[1].block)
    J = Block(JR)
    K ∈ blockcolsupport(A, J) || throw(ArgumentError("Pointer is only defined when inside blockcolsupport"))
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
