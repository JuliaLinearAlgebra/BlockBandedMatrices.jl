####
# Matrix memory layout traits
#
# if MemoryLayout(A) returns BandedColumnMajor, you must override
# pointer and leadingdimension
# in addition to the banded matrix interface
####

abstract type AbstractBlockBandedLayout <: MemoryLayout end

struct BandedBlockBandedColumnMajor <: AbstractBlockBandedLayout end
struct BandedBlockBandedRowMajor <: AbstractBlockBandedLayout end
struct BlockBandedColumnMajor <: AbstractBlockBandedLayout end
struct BlockBandedRowMajor <: AbstractBlockBandedLayout end


transposelayout(::BandedBlockBandedColumnMajor) = BandedBlockBandedRowMajor()
transposelayout(::BandedBlockBandedRowMajor) = BandedBlockBandedColumnMajor()
transposelayout(::BlockBandedColumnMajor) = BlockBandedRowMajor()
transposelayout(::BlockBandedRowMajor) = BlockBandedColumnMajor()
conjlayout(::Type{<:Complex}, M::AbstractBlockBandedLayout) = ConjLayout(M)

# Here we override broadcasting for banded matrices.
# The design is to to exploit the broadcast machinery so that
# banded matrices that conform to the banded matrix interface but are not
# <: AbstractBandedMatrix can get access to fast copyto!, lmul!, rmul!, axpy!, etc.
# using broadcast variants (B .= A, B .= 2.0 .* A, etc.)


abstract type AbstractBlockBandedStyle <: AbstractArrayStyle{2} end
struct BandedBlockBandedStyle <: AbstractBlockBandedStyle end
abstract type AbstractBlockSkylineStyle <: AbstractBlockBandedStyle end
struct BlockSkylineStyle <: AbstractBlockSkylineStyle end
struct BlockBandedStyle <: AbstractBlockSkylineStyle end
BlockSkylineStyle(::Val{2}) = BlockBandedStyle()
BlockBandedStyle(::Val{2}) = BlockBandedStyle()
BandedBlockBandedStyle(::Val{2}) = BandedBlockBandedStyle()
BroadcastStyle(::DefaultArrayStyle{2}, ::AbstractBlockBandedStyle) = DefaultArrayStyle{2}()
BroadcastStyle(::AbstractBlockBandedStyle, ::DefaultArrayStyle{2}) = DefaultArrayStyle{2}()

BroadcastStyle(::BlockSkylineStyle, ::BandedBlockBandedStyle) = BlockSkylineStyle()
BroadcastStyle(::BandedBlockBandedStyle, ::BlockSkylineStyle) = BlockSkylineStyle()
BroadcastStyle(::BlockSkylineStyle, ::BlockBandedStyle) = BlockSkylineStyle()
BroadcastStyle(::BlockBandedStyle, ::BlockSkylineStyle) = BlockSkylineStyle()

BroadcastStyle(::BlockBandedStyle, ::BandedBlockBandedStyle) = BlockBandedStyle()
BroadcastStyle(::BandedBlockBandedStyle, ::BlockBandedStyle) = BlockBandedStyle()



####
# Default to standard Array broadcast
#
# This is because, for example, exp.(B) is not banded
####


copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractBlockBandedStyle}) =
   copyto!(dest, Broadcasted{DefaultArrayStyle{2}}(bc.f, bc.args, bc.axes))

similar(bc::Broadcasted{<:AbstractBlockBandedStyle}, ::Type{T}) where T =
    similar(Broadcasted{DefaultArrayStyle{2}}(bc.f, bc.args, bc.axes), T)

##
# copyto!
##

function checkblocks(A, B)
    Arows, Acols = cumulsizes(A)
    Brows, Bcols = cumulsizes(B)
    if Acols ≠ Bcols || Arows ≠ Brows
        throw(DimensionMismatch("*"))
    end
end

function _blockbanded_copyto!(dest::AbstractMatrix{T}, src::AbstractMatrix) where T
    @boundscheck checkblocks(dest, src)

    dl, du = colblockbandwidths(dest)
    sl, su = colblockbandwidths(src)
    M,N = nblocks(src)
    # Source matrix must fit within bands of destination matrix
    all(dl .≥ min.(sl,Ref(M-1))) && all(du .≥ min.(su,Ref(N-1))) || throw(BandError(dest))

    for J = 1:N
        for K = max(1,J-du[J]):min(J-su[J]-1,M)
            view(dest,Block(K),Block(J)) .= zero(T)
        end
        for K = max(1,J-su[J]):min(J+sl[J],M)
            view(dest,Block(K),Block(J)) .= view(src,Block(K),Block(J))
        end
        for K = max(1,J+sl[J]+1):min(J+dl[J],M)
            view(dest,Block(K),Block(J)) .= zero(T)
        end
    end
    dest
end

function blockbanded_copyto!(dest::AbstractMatrix, src::AbstractMatrix)
    if isblockbanded(dest)
        _blockbanded_copyto!(dest, src)
    else
        _blockbanded_copyto!(PseudoBlockArray(dest, blocksizes(src).block_sizes), src)
    end
    dest
end


copyto!(dest::AbstractMatrix, src::AbstractBlockBandedMatrix) = blockbanded_copyto!(dest, src)


function copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(identity)})
    (A,) = bc.args
    dest ≡ A && return dest
    blockbanded_copyto!(dest, A)
end


##
# lmul!/rmul!
##

######
# extra marrix routines
#####

function blockbanded_fill!(B::AbstractMatrix{T}, x) where T
    x == zero(T) || throw(BandError(B))

    M,N = nblocks(B)
    for J = 1:N, K = blockcolrange(B,J)
        fill!(view(B,K,Block(J)), x)
    end
    B
end


function blockbanded_rmul!(B::AbstractMatrix{T}, x::Number) where T
    x == zero(T) || throw(BandError(B))

    M,N = nblocks(B)
    for J = 1:N, K = blockcolrange(B,J)
        rmul!(view(B,K,Block(J)), x)
    end
    B
end

function blockbanded_lmul!(x::Number, B::AbstractMatrix{T}) where T
    x == zero(T) || throw(BandError(B))

    M,N = nblocks(B)
    for J = 1:N, K = blockcolrange(B,J)
        lmul!(x, view(B,K,Block(J)))
    end
    B
end


function blockbanded_fill!(B::AbstractBlockBandedMatrix, x::Number)
    x == zero(T) || throw(BandError(B))

    fill!(B.data, x)
end

function blockbanded_rmul!(A::AbstractBlockBandedMatrix, x::Number)
    rmul!(A.data, x)
    A
end


function blockbanded_lmul!(x::Number, A::AbstractBlockBandedMatrix)
    lmul!(x, A.data)
    A
end

lmul!(x::Number, A::AbstractBlockBandedMatrix) = blockbanded_lmul!(x, A)
rmul!(A::AbstractBlockBandedMatrix, x::Number) = blockbanded_rmul!(A, x)



function copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(-), <:Tuple{<:AbstractMatrix}})
    A, = bc.args
    dest ≡ A || copyto!(dest, A)
    blockbanded_lmul!(-1, dest)
end

similar(bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(-), <:Tuple{<:AbstractMatrix}}, ::Type{T}) where T =
    similar(bc.args[1], T)

function copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(*), <:Tuple{<:Number,<:AbstractMatrix}})
    α,A = bc.args
    dest ≡ A || copyto!(dest, A)
    blockbanded_lmul!(α, dest)
end

similar(bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(*), <:Tuple{<:Number,<:AbstractMatrix}}, ::Type{T}) where T =
    similar(bc.args[2], T)


function copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(*), <:Tuple{<:AbstractMatrix,<:Number}})
    A,α = bc.args
    dest ≡ A || copyto!(dest, A)
    blockbanded_rmul!(dest, α)
end

similar(bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(*), <:Tuple{<:AbstractMatrix,<:Number}}, ::Type{T}) where T =
    similar(bc.args[1], T)

function copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(/), <:Tuple{<:AbstractMatrix,<:Number}})
    A,α = bc.args
    dest ≡ A || copyto!(dest, A)
    blockbanded_rmul!(dest, inv(α))
end

similar(bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(/), <:Tuple{<:AbstractMatrix,<:Number}}, ::Type{T}) where T =
    similar(bc.args[1], T)

function copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(\), <:Tuple{<:Number,<:AbstractMatrix}})
    α,A = bc.args
    dest ≡ A || copyto!(dest, A)
    blockbanded_lmul!(inv(α), dest)
end

similar(bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(\), <:Tuple{<:Number,<:AbstractMatrix}}, ::Type{T}) where T =
    similar(bc.args[2], T)

##
# axpy!
##

# these are the routines of the banded interface of other AbstractMatrices
function blockbanded_axpy!(a, X::AbstractMatrix, Y::AbstractMatrix)
    size(X) == size(Y) || throw(DimensionMismatch())

    for J=Block(1):Block(nblocks(X,2)), K=blockcolrange(X,J)
        view(Y,K,J) .= a .* view(X,K,J) .+ view(Y,K,J)
    end
    Y
end

function _combine_blocksizes(A, B)
    A_size = blocksizes(A).block_sizes
    A_size == blocksizes(B).block_sizes || throw(DimensionMismatch())
    A_size
end

_combine_blocksizes(::Diagonal, B) = blocksizes(B)
_combine_blocksizes(A, ::Diagonal) = blocksizes(A)

function _combined_blocksizes(A, B)
    blocksizes(A) == blocksizes(B) || throw(DimensionMismatch("Block sizes do not agree"))
    (A,B)
end

_combined_blocksizes(A::Diagonal, B) = PseudoBlockArray(A, blocksizes(B).block_sizes), B
_combined_blocksizes(A, B::Diagonal) = A, PseudoBlockArray(B, blocksizes(A).block_sizes)



for op in (:+, :-)
    @eval begin
        function similar(bc::Broadcasted{<:BlockSkylineStyle, <:Any, typeof($op), <:Tuple{<:AbstractMatrix,<:AbstractMatrix}}, ::Type{T}) where T
            A,B = bc.args
            Al,Au = colblockbandwidths(A)
            Bl,Bu = colblockbandwidths(B)
            BlockSkylineMatrix{T}(undef, _combine_blocksizes(A,B), (max.(Al,Bl), max.(Au,Bu)))
        end

        function similar(bc::Broadcasted{<:BlockBandedStyle, <:Any, typeof($op), <:Tuple{<:AbstractMatrix,<:AbstractMatrix}}, ::Type{T}) where T
            A,B = bc.args
            Al,Au = blockbandwidths(A)
            Bl,Bu = blockbandwidths(B)
            BlockBandedMatrix{T}(undef, _combine_blocksizes(A,B), (max(Al,Bl), max(Au,Bu)))
        end

        function similar(bc::Broadcasted{<:BandedBlockBandedStyle, <:Any, typeof($op), <:Tuple{<:AbstractMatrix,<:AbstractMatrix}}, ::Type{T}) where T
            A,B = bc.args
            Al,Au = blockbandwidths(A)
            Bl,Bu = blockbandwidths(B)
            Aλ,Aμ = subblockbandwidths(A)
            Bλ,Bμ = subblockbandwidths(B)

            BandedBlockBandedMatrix{T}(undef, _combine_blocksizes(A,B), (max(Al,Bl), max(Au,Bu)), (max(Aλ,Bλ), max(Aμ,Bμ)))
        end
    end
end

for op in (:+, :-)
    @eval function copyto!(C::AbstractArray{T}, bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof($op),
                                                                <:Tuple{<:AbstractMatrix,<:AbstractMatrix}}) where T
        A,B = _combined_blocksizes(bc.args...)
        A_l,A_u = blockbandwidths(A)
        B_l,B_u = blockbandwidths(B)
        C_l,C_u = blockbandwidths(C)

        size(A) == size(B) == size(C) || throw(DimensionMismatch())
        N,M = nblocks(C)

        for J̃ = 1:M
            J = Block(J̃)
            for K = Block.(max(1,J̃-min(A_u,B_u)):min(N,J̃+min(A_l,B_l)))
                view(C,K,J) .= $op.(view(A,K,J), view(B,K,J))
            end
            for K = Block.(max(1,J̃-A_u):min(N,J̃-B_u-1))
                view(C,K,J) .= view(A,K,J)
            end
            for K = Block.(max(1,J̃-B_u):min(N,J̃-A_u-1))
                view(C,K,J) .= $op.(view(B,K,J))
            end
            for K = Block.(max(1,J̃+B_l+1):min(N,J̃+A_u))
                view(C,K,J) .= view(A,K,J)
            end
            for K = Block.(max(1,J̃+A_l+1):min(N,J̃+B_u))
                view(C,K,J) .= $op.(view(B,K,J))
            end
            for K = Block.(max(J̃-C_u,1):min(J̃-max(A_u,B_u)-1,N))
                view(C,K,J) .= zero(T)
            end
            for K = Block.(max(J̃+max(A_l,B_l)+1,1):min(J̃+C_u,N))
                view(C,K,J) .= zero(T)
            end
        end

        C
    end
end


function copyto!(dest::AbstractArray{T}, bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof(+),
                                                        <:Tuple{<:Broadcasted{<:AbstractBlockBandedStyle,<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}},
                                                        <:AbstractMatrix}}) where T
    αA,B = bc.args
    α,A = αA.args
    dest ≡ B || blockbanded_copyto!(dest, B)
    blockbanded_axpy!(α, A, dest)
end

function similar(bc::Broadcasted{BlockBandedStyle, <:Any, typeof(+),
                        <:Tuple{<:Broadcasted{<:AbstractBlockBandedStyle,<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}},
                        <:AbstractMatrix}}, ::Type{T}) where T
    αA,B = bc.args
    α,A = αA.args
    Al,Au = blockbandwidths(A)
    Bl,Bu = blockbandwidths(B)
    BlockBandedMatrix{T}(undef, _combine_blocksizes(A,B), (max(Al,Bl), max(Au,Bu)))
end

function similar(bc::Broadcasted{BandedBlockBandedStyle, <:Any, typeof(+),
                        <:Tuple{<:Broadcasted{<:AbstractBlockBandedStyle,<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}},
                        <:AbstractMatrix}}, ::Type{T}) where T
    αA,B = bc.args
    α,A = αA.args
    Al,Au = blockbandwidths(A)
    Bl,Bu = blockbandwidths(B)
    Aλ,Aμ = subblockbandwidths(A)
    Bλ,Bμ = subblockbandwidths(B)

    BandedBlockBandedMatrix{T}(undef, _combine_blocksizes(A,B), (max(Al,Bl), max(Au,Bu)), (max(Aλ,Bλ), max(Aμ,Bμ)))
end
