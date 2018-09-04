@lazylmul UpperTriangular{T, BandedBlockBandedMatrix{T}} where T
@lazylmul UnitUpperTriangular{T, BandedBlockBandedMatrix{T}} where T
@lazylmul LowerTriangular{T, BandedBlockBandedMatrix{T}} where T
@lazylmul UnitLowerTriangular{T, BandedBlockBandedMatrix{T}} where T


@lazyldiv UpperTriangular{T, BandedBlockBandedMatrix{T}} where T
@lazyldiv UnitUpperTriangular{T, BandedBlockBandedMatrix{T}} where T
@lazyldiv LowerTriangular{T, BandedBlockBandedMatrix{T}} where T
@lazyldiv UnitLowerTriangular{T, BandedBlockBandedMatrix{T}} where T


@inline hasmatchingblocks(A) =
    cumulsizes(blocksizes(A),1) == cumulsizes(blocksizes(A),2)


#TODO: non-matchin g blocks
isblockbanded(A::AbstractTriangular) =
    isblockbanded(parent(A)) && hasmatchingblocks(A)
isbandedblockbanded(A::AbstractTriangular) =
    isbandedblockbanded(parent(A)) && hasmatchingblocks(A)
blockbandwidths(A::Union{UpperTriangular,UnitUpperTriangular}) = let P = parent(A)
        (min(0,blockbandwidths(P,1)), blockbandwidth(P,2))
    end
blockbandwidths(A::Union{LowerTriangular,UnitLowerTriangular}) = let P = parent(A)
        (blockbandwidth(P,1), min(0,blockbandwidths(P,2)))
    end
subblockbandwidths(A::AbstractTriangular) = subblockbandwidths(parent(A))

triangularlayout(::Type{Tri}, ML::BandedBlockBandedColumnMajor) where {Tri} = Tri(ML)


function upper_blockbanded_squareblocks_trtrs!(A::AbstractMatrix{T}, b_in::AbstractVector{T}) where T
    @boundscheck size(A,1) == size(b_in,1) || throw(BoundsError())

    # impose block structure
    b = PseudoBlockArray(b_in, BlockSizes((cumulsizes(blocksizes(A),1),)))

    Bs = blocksizes(A)
    N = nblocks(Bs,1)

    for K = N:-1:1
        b_2 = view(b, Block(K))
        b_2 .= Ldiv(UpperTriangular(view(A, Block(K),  Block(K))), b_2)

        if K ≥ 2
            V_12 = view(A, blockcolstart(A, K):Block(K-1), Block(K))
            b̃_1 = view(b, blockcolstart(A, K):Block(K-1))
            b̃_1 .=  (-one(T)).*Mul(V_12, b_2) .+ b̃_1
        end
    end

    b
end

function upper_blockbanded_squareblocks_mul!(A::AbstractMatrix{T}, b_in::AbstractVector{T}) where T
    @boundscheck size(A,1) == size(b_in,1) || throw(BoundsError())

    # impose block structure
    b = PseudoBlockArray(b_in, BlockSizes((cumulsizes(blocksizes(A),1),)))

    Bs = blocksizes(A)
    N = nblocks(Bs,1)

    for K = 1:N
        b_2 = view(b, Block(K))
        b_2 .= Mul(UpperTriangular(view(A, Block(K,K))), b_2)
        JR = Block(K+1):blockrowstop(A,K)
        if !isempty(JR)
            b_2 .= Mul(view(A, Block(K), JR), view(b,JR)) .+ b_2
        end
    end

    b_in
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatMulVec{T, <:TriangularLayout{'U',UNIT,BandedBlockBandedColumnMajor},
                                   <:AbstractStridedLayout}) where {UNIT,T}
    A,x = M.A, M.B
    @assert hasmatchingblocks(A)
    x ≡ dest || copyto!(dest, x)
    upper_blockbanded_squareblocks_mul!(triangulardata(A), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatLdivVec{T, <:TriangularLayout{'U',UNIT,BandedBlockBandedColumnMajor},
                                   <:AbstractStridedLayout}) where {UNIT,T}
    A,x = inv(M.A), M.B
    @assert hasmatchingblocks(A)
    x ≡ dest || copyto!(dest, x)
    upper_blockbanded_squareblocks_trtrs!(triangulardata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatLdivVec{T, <:TriangularLayout{'L',UNIT,BandedBlockBandedColumnMajor},
                                   <:AbstractStridedLayout}) where {UNIT,T}
    A,x = inv(M.A), M.B
    @assert hasmatchingblocks(A)
    x ≡ dest || copyto!(dest, x)
    lower_blockbanded_squareblocks_trtrs!(triangulardata(A), dest)
end







##################
# UpperBandedBlockBandedBlock
#
#   views of the blocks satisfy the BandedMatrix interface, and are memory-compatible
#   with BLASBandedMatrix.
##################

const UpperBandedBlockBandedBlock{T} = SubArray{T,2,UpperTriangular{T,BandedBlockBandedMatrix{T}},Tuple{BlockSlice1,BlockSlice1},false}


isbanded(::UpperBandedBlockBandedBlock) = true
# Not type stable, but infers Union type so should be "fast"
MemoryLayout(B::UpperBandedBlockBandedBlock) =
    ==(parentindices(B)...) ? UpperTriangularLayout(BandedColumnMajor()) : BandedColumnMajor()

function inblockbands(V::UpperBandedBlockBandedBlock)
    A = parent(V)
    K_sl, J_sl = parentindices(V)
    K, J = K_sl.block, J_sl.block
    -A.l ≤ Int(J-K) ≤ A.u
end


######################################
# BandedMatrix interface  for Blocks #
######################################
@inline bandwidths(V::UpperBandedBlockBandedBlock) = subblockbandwidths(parent(V))



# gives the columns of parent(V).data that encode the block
blocks(V::UpperBandedBlockBandedBlock)::Tuple{Int,Int} = Int(first(parentindices(V)).block),
                                                    Int(last(parentindices(V)).block)


function tribandeddata(V::UpperBandedBlockBandedBlock)
    A = parent(parent(V))
    K_sl, J_sl = parentindices(V)
    K, J = K_sl.block, J_sl.block
    tribandeddata(UpperTriangular(view(A, K, J)))
end


@inline inbands_getindex(V::UpperBandedBlockBandedBlock, k::Int, j::Int) =
    banded_getindex(bandeddata(V), bandwidths(V)..., k, j)

@inline inbands_setindex!(V::UpperBandedBlockBandedBlock, v, k::Int, j::Int) =
    banded_setindex!(bandeddata(V), bandwidths(V)..., v, k, j)
