@lazylmul UpperTriangular{T, DefaultBandedBlockBandedMatrix{T}} where T
@lazylmul UnitUpperTriangular{T, DefaultBandedBlockBandedMatrix{T}} where T
@lazylmul LowerTriangular{T, DefaultBandedBlockBandedMatrix{T}} where T
@lazylmul UnitLowerTriangular{T, DefaultBandedBlockBandedMatrix{T}} where T


@lazyldiv UpperTriangular{T, DefaultBandedBlockBandedMatrix{T}} where T
@lazyldiv UnitUpperTriangular{T, DefaultBandedBlockBandedMatrix{T}} where T
@lazyldiv LowerTriangular{T, DefaultBandedBlockBandedMatrix{T}} where T
@lazyldiv UnitLowerTriangular{T, DefaultBandedBlockBandedMatrix{T}} where T


@inline hasmatchingblocks(A) =
    cumulsizes(blocksizes(A),1) == cumulsizes(blocksizes(A),2)


#TODO: non-matchin g blocks
isblockbanded(A::AbstractTriangular) =
    isblockbanded(parent(A))
isbandedblockbanded(A::AbstractTriangular) =
    isbandedblockbanded(parent(A))
blockbandwidths(A::Union{UpperTriangular,UnitUpperTriangular}) = let P = parent(A)
        if hasmatchingblocks(P)
            (min(0,blockbandwidths(P,1)), blockbandwidth(P,2))
        else
            blockbandwidths(P)
        end
    end
blockbandwidths(A::Union{LowerTriangular,UnitLowerTriangular}) = let P = parent(A)
        if hasmatchingblocks(P)
            (blockbandwidth(P,1), min(0,blockbandwidth(P,2)))
        else
            blockbandwidths(P)
        end
    end
subblockbandwidths(A::AbstractTriangular) = subblockbandwidths(parent(A))

triangularlayout(::Type{Tri}, ML::BandedBlockBandedColumnMajor) where {Tri} = Tri(ML)

_triangular_matrix(::Val{'U'}, ::Val{'N'}, A) = UpperTriangular(A)
_triangular_matrix(::Val{'L'}, ::Val{'N'}, A) = LowerTriangular(A)
_triangular_matrix(::Val{'U'}, ::Val{'U'}, A) = UnitUpperTriangular(A)
_triangular_matrix(::Val{'L'}, ::Val{'U'}, A) = UnitLowerTriangular(A)


function _matchingblocks_triangular_mul!(::Val{'U'}, UNIT, A, dest)
    # impose block structure
    b = PseudoBlockArray(dest, BlockSizes((cumulsizes(blocksizes(A),1),)))

    Bs = blocksizes(A)
    N = nblocks(Bs,1)

    for K = 1:N
        b_2 = view(b, Block(K))
        Ũ = _triangular_matrix(Val('U'), UNIT, view(A, Block(K,K)))
        b_2 .= Mul(Ũ, b_2)
        JR = Block(K+1):blockrowstop(A,K)
        if !isempty(JR)
            b_2 .= Mul(view(A, Block(K), JR), view(b,JR)) .+ b_2
        end
    end
    dest
end

function _matchingblocks_triangular_mul!(::Val{'L'}, UNIT, A, dest)
    # impose block structure
    b = PseudoBlockArray(dest, BlockSizes((cumulsizes(blocksizes(A),1),)))

    Bs = blocksizes(A)
    N = nblocks(Bs,1)

    for K = N:-1:1
        b_2 = view(b, Block(K))
        L̃ = _triangular_matrix(Val('L'), UNIT, view(A, Block(K,K)))
        b_2 .= Mul(L̃, b_2)
        JR = blockrowstart(A,K):Block(K-1)
        if !isempty(JR)
            b_2 .= Mul(view(A, Block(K), JR), view(b,JR)) .+ b_2
        end
    end

    dest
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{<:TriangularLayout{UPLO,UNIT,LAY},
                                   <:AbstractStridedLayout,T,T}) where {UPLO,UNIT,T<:BlasFloat,LAY<:BandedBlockBandedColumnMajor}
    U,x = M.factors
    @boundscheck size(U,1) == size(dest,1) || throw(BoundsError())
    if hasmatchingblocks(U)
        x ≡ dest || copyto!(dest, x)
        _matchingblocks_triangular_mul!(Val(UPLO), Val(UNIT), triangulardata(U), dest)
    else # use default
        if x ≡ dest
            blasmul!(dest, U, copy(x), one(T), zero(T), MemoryLayout(dest), BandedBlockBandedColumnMajor(), MemoryLayout(x))
        else
            blasmul!(dest, U, x, one(T), zero(T), MemoryLayout(dest), BandedBlockBandedColumnMajor(), MemoryLayout(x))
        end
    end
end



@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'U',UNIT,BandedBlockBandedColumnMajor},
                                   <:AbstractStridedLayout}) where {T,UNIT}
    Ui,x = M.factors
    U = inv(Ui)
    x ≡ dest || copyto!(dest, x)
    A = triangulardata(U)
    @assert hasmatchingblocks(A)

    @boundscheck size(A,1) == size(dest,1) || throw(BoundsError())

    # impose block structure
    b = PseudoBlockArray(dest, BlockSizes((cumulsizes(blocksizes(A),1),)))

    Bs = blocksizes(A)
    N = nblocks(Bs,1)

    for K = N:-1:1
        b_2 = view(b, Block(K))
        Ũ = _triangular_matrix(Val('U'), Val(UNIT), view(A, Block(K,K)))
        b_2 .= Ldiv(Ũ, b_2)

        if K ≥ 2
            KR = blockcolstart(A, K):Block(K-1)
            V_12 = view(A, KR, Block(K))
            b̃_1 = view(b, KR)
            b̃_1 .=  (-one(T)).*Mul(V_12, b_2) .+ b̃_1
        end
    end

    dest
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'L',UNIT,BandedBlockBandedColumnMajor},
                                   <:AbstractStridedLayout}) where {UNIT,T}
    Li,x = M.factors
    L = inv(Li)
    x ≡ dest || copyto!(dest, x)
    A = triangulardata(L)
    @assert hasmatchingblocks(A)

    @boundscheck size(A,1) == size(dest,1) || throw(BoundsError())

    # impose block structure
    b = PseudoBlockArray(dest, BlockSizes((cumulsizes(blocksizes(A),1),)))

    Bs = blocksizes(A)
    N = nblocks(Bs,1)

    for K = 1:N
       b_2 = view(b, Block(K))
       L̃ = _triangular_matrix(Val('L'), Val(UNIT), view(A, Block(K,K)))
       b_2 .= Ldiv(L̃, b_2)

       if K < N
           KR = Block(K+1):blockcolstop(A, K)
           V_12 = view(A, KR, Block(K))
           b̃_1 = view(b, KR)
           b̃_1 .=  (-one(T)).*Mul(V_12, b_2) .+ b̃_1
       end
    end

    dest
end







##################
# UpperBandedBlockBandedBlock
#
#   views of the blocks satisfy the BandedMatrix interface, and are memory-compatible
#   with BLASBandedMatrix.
##################

# const UpperBandedBlockBandedBlock{T} = SubArray{T,2,UpperTriangular{T,BandedBlockBandedMatrix{T}},Tuple{BlockSlice1,BlockSlice1},false}
#
#
# isbanded(::UpperBandedBlockBandedBlock) = true
# # Not type stable, but infers Union type so should be "fast"
# MemoryLayout(B::UpperBandedBlockBandedBlock) =
#     ==(parentindices(B)...) ? UpperTriangularLayout(BandedColumnMajor()) : BandedColumnMajor()
#
# function inblockbands(V::UpperBandedBlockBandedBlock)
#     A = parent(V)
#     K_sl, J_sl = parentindices(V)
#     K, J = K_sl.block, J_sl.block
#     -A.l ≤ Int(J-K) ≤ A.u
# end
#
#
# ######################################
# # BandedMatrix interface  for Blocks #
# ######################################
# @inline bandwidths(V::UpperBandedBlockBandedBlock) = subblockbandwidths(parent(V))
#
#
#
# # gives the columns of parent(V).data that encode the block
# blocks(V::UpperBandedBlockBandedBlock)::Tuple{Int,Int} = Int(first(parentindices(V)).block),
#                                                     Int(last(parentindices(V)).block)
#
#
# function tribandeddata(V::UpperBandedBlockBandedBlock)
#     A = parent(parent(V))
#     K_sl, J_sl = parentindices(V)
#     K, J = K_sl.block, J_sl.block
#     tribandeddata(UpperTriangular(view(A, K, J)))
# end
#
#
# @inline inbands_getindex(V::UpperBandedBlockBandedBlock, k::Int, j::Int) =
#     banded_getindex(bandeddata(V), bandwidths(V)..., k, j)
#
# @inline inbands_setindex!(V::UpperBandedBlockBandedBlock, v, k::Int, j::Int) =
#     banded_setindex!(bandeddata(V), bandwidths(V)..., v, k, j)
