
# Here we override broadcasting for banded matrices.
# The design is to to exploit the broadcast machinery so that
# banded matrices that conform to the banded matrix interface but are not
# <: AbstractBandedMatrix can get access to fast copyto!, lmul!, rmul!, axpy!, etc.
# using broadcast variants (B .= A, B .= 2.0 .* A, etc.)


abstract type AbstractBlockBandedStyle <: AbstractArrayStyle{2} end
struct BandedBlockBandedStyle <: AbstractBlockBandedStyle end
abstract type AbstractBlockSkylineStyle <: AbstractBlockBandedStyle end
struct BlockSkylineStyle <: AbstractBlockSkylineStyle end
struct BlockBandedStyle <: AbstractBlockSkylineStyle end
BlockSkylineStyle(::Val{2}) = BlockBandedStyle()
BlockBandedStyle(::Val{2}) = BlockBandedStyle()
BandedBlockBandedStyle(::Val{2}) = BandedBlockBandedStyle()

BroadcastStyle(::BlockSkylineStyle, ::BandedBlockBandedStyle) = BlockSkylineStyle()
BroadcastStyle(::BandedBlockBandedStyle, ::BlockSkylineStyle) = BlockSkylineStyle()
BroadcastStyle(::BlockSkylineStyle, ::BlockBandedStyle) = BlockSkylineStyle()
BroadcastStyle(::BlockBandedStyle, ::BlockSkylineStyle) = BlockSkylineStyle()

BroadcastStyle(::BlockBandedStyle, ::BandedBlockBandedStyle) = BlockBandedStyle()
BroadcastStyle(::BandedBlockBandedStyle, ::BlockBandedStyle) = BlockBandedStyle()

BroadcastStyle(::DefaultArrayStyle{2}, ::BlockSkylineStyle) = BlockSkylineStyle()
BroadcastStyle(::BlockSkylineStyle, ::DefaultArrayStyle{2}) = BlockSkylineStyle()

BroadcastStyle(::DefaultArrayStyle{2}, ::BlockBandedStyle) = BlockBandedStyle()
BroadcastStyle(::BlockBandedStyle, ::DefaultArrayStyle{2}) = BlockBandedStyle()

BroadcastStyle(::DefaultArrayStyle{2}, ::BandedBlockBandedStyle) = BandedBlockBandedStyle()
BroadcastStyle(::BandedBlockBandedStyle, ::DefaultArrayStyle{2}) = BandedBlockBandedStyle()

###
# broadcast blockbandwidths
###
_blockbnds(bc) = blocksize(bc) .- 1

_blockbandwidth_u(u, a, ax) = blockisequal(axes(a,1),ax) ? (blockbandwidth(a,1),u) : (blocksize(ax,1)-1,u)
_blockbandwidth_l(l, a, ax) = blockisequal(axes(a,2),ax) ? (l,blockbandwidth(a,2)) : (l,blocksize(ax,1)-1)

_broadcast_blockbandwidths(bnds, _, _) = bnds
_broadcast_blockbandwidths((l,u), a::AbstractVector, (ax1,ax2)) =
    _blockbandwidth_u(u, a, ax1)


function _broadcast_blockbandwidths((l,u), A::AbstractArray, (ax1,ax2))
    if size(A,2) == 1
        _blockbandwidth_u(u, A, ax1)
    elseif size(A,1) == 1
        _blockbandwidth_l(l, A, ax2)
    elseif blockisequal(axes(A), (ax1,ax2))
        blockbandwidths(A) # need to special case vector broadcasting
    else
        _blockbnds(A)
    end
end


blockbandwidths(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Any,typeof(*)}) =
    min.(_broadcast_blockbandwidths.(Ref(_blockbnds(bc)), bc.args, Ref(axes(bc)))...)

blockbandwidths(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Any,typeof(/)}) = _broadcast_blockbandwidths(_blockbnds(bc), first(bc.args), axes(bc))
blockbandwidths(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Any,typeof(\)}) = _broadcast_blockbandwidths(_blockbnds(bc), last(bc.args), axes(bc))

_subblockbnds(bc) = size(bc) .- 1

_broadcast_subblockbandwidths(bnds) = bnds
_broadcast_subblockbandwidths(bnds, _) = bnds
_broadcast_subblockbandwidths((l,u), a::AbstractVector) = (bandwidth(a,1),u)

function _broadcast_subblockbandwidths((l,u), A::AbstractArray)
    size(A,2) == 1 && return (subblockbandwidth(A,1),u)
    size(A,1) == 1 && return (l, subblockbandwidth(A,2))
    subblockbandwidths(A) # need to special case vector broadcasting
end

subblockbandwidths(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Any,typeof(*)}) =
    min.(_broadcast_subblockbandwidths.(Ref(_subblockbnds(bc)), bc.args)...)

subblockbandwidths(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Any,typeof(/)}) = _broadcast_subblockbandwidths(_subblockbnds(bc), first(bc.args))
subblockbandwidths(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Any,typeof(\)}) = _broadcast_subblockbandwidths(_subblockbnds(bc), last(bc.args))

function subblockbandwidths(bc::Broadcasted)
    (a,b) = size(bc)
    bnds = (a-1,b-1)
    _isweakzero(bc.f, bc.args...) && return min.(bnds, max.(_broadcast_subblockbandwidths.(Ref(bnds), bc.args)...))
    bnds
end


# zero is preserved. Take the maximum bandwidth
import BandedMatrices: _isweakzero

function blockbandwidths(bc::Broadcasted)
    (a,b) = size(bc)
    bnds = (a-1,b-1)
    _isweakzero(bc.f, bc.args...) && return min.(bnds, max.(_broadcast_blockbandwidths.(Ref(bnds), bc.args, Ref(axes(bc)))...))
    bnds
end

similar(bc::Broadcasted{BlockBandedStyle}, ::Type{T}) where T = BlockBandedMatrix{T}(undef, axes(bc), blockbandwidths(bc))
similar(bc::Broadcasted{BandedBlockBandedStyle}, ::Type{T}) where T = BandedBlockBandedMatrix{T}(undef, axes(bc), blockbandwidths(bc), subblockbandwidths(bc))


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

checkblocks(A, B) = blockisequal(axes(A), axes(B)) || throw(DimensionMismatch("*"))

function _blockbanded_copyto!(dest::AbstractMatrix, src::AbstractMatrix)
    @boundscheck checkblocks(dest, src)

    dl, du = colblockbandwidths(dest)
    sl, su = colblockbandwidths(src)
    M,N = blocksize(src)
    # Source matrix must fit within bands of destination matrix
    all(dl .≥ min.(sl,Ref(M-1))) && all(du .≥ min.(su,Ref(N-1))) || throw(BandError(dest))

    for J = 1:N
        for K = max(1,J-du[J]):min(J-su[J]-1,M)
            zero!(view(dest,Block(K),Block(J)))
        end
        for K = max(1,J-su[J]):min(J+sl[J],M)
            copyto!(view(dest,Block(K),Block(J)), view(src,Block(K),Block(J)))
        end
        for K = max(1,J+sl[J]+1):min(J+dl[J],M)
            zero!(view(dest,Block(K),Block(J)))
        end
    end
    dest
end

function blockbanded_copyto!(dest::AbstractMatrix, src::AbstractMatrix)
    if isblockbanded(dest)
        _blockbanded_copyto!(dest, src)
    else
        _blockbanded_copyto!(BlockedArray(dest, axes(src)), src)
    end
    dest
end


_copyto!(_, ::AbstractBlockBandedLayout, dest::AbstractMatrix, src::AbstractMatrix) = blockbanded_copyto!(dest, src)


function _copyto!(::BandedBlockBandedColumns, ::BandedBlockBandedColumns, dest::AbstractMatrix, src::AbstractMatrix)
    if blockbandwidths(dest) == blockbandwidths(src) && subblockbandwidths(dest) == subblockbandwidths(src)
        copyto!(bandedblockbandeddata(dest), bandedblockbandeddata(src))
        dest
    else # TODO: if subblockbandwidths match we can still do something better
        _copyto!(BandedBlockBandedLayout(), BandedBlockBandedLayout(), dest, src)
    end
end


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

    for J = blockaxes(B,2), K = blockcolsupport(B,J)
        fill!(view(B,K,J), x)
    end
    B
end


function blockbanded_rmul!(B::AbstractMatrix{T}, x::Number) where T
    for J = blockaxes(B,2), K = blockcolsupport(B,J)
        rmul!(view(B,K,J), x)
    end
    B
end

function blockbanded_lmul!(x::Number, B::AbstractMatrix{T}) where T
    for J = blockaxes(B,2), K = blockcolsupport(B,J)
        lmul!(x, view(B,K,J))
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

    for J = blockaxes(X,2), K = blockcolsupport(X,J)
        view(Y,K,J) .= a .* view(X,K,J) .+ view(Y,K,J)
    end
    Y
end


for op in (:+, :-)
    @eval function copyto!(C::AbstractArray{T}, bc::Broadcasted{<:AbstractBlockBandedStyle, <:Any, typeof($op),
                                                                <:Tuple{<:AbstractMatrix,<:AbstractMatrix}}) where T
        bc_axes = Base.Broadcast.combine_axes(bc.args...)# Use combine_axes as `bc` might get axes from `C`
        if !blockisequal(axes(C), bc_axes)
            copyto!(BlockedArray(C, bc_axes), bc)
            return C
        end

        A,B = bc.args
        if !blockisequal(axes(A), axes(B))
            copyto!(C, Base.broadcasted(BlockStyle{2}(), $op, A, B))
            return C
        end

        A_l,A_u = blockbandwidths(A)
        B_l,B_u = blockbandwidths(B)
        C_l,C_u = blockbandwidths(C)

        N,M = blocksize(C)

        for J̃ = 1:M
            J = Block(J̃)
            # joint block range
            for K = Block.(max(1,J̃-min(A_u,B_u)):min(N,J̃+min(A_l,B_l)))
                view(C,K,J) .= $op.(view(A,K,J), view(B,K,J))
            end
            # above, A but not B
            for K = Block.(max(1,J̃-A_u):min(N,J̃-B_u-1))
                view(C,K,J) .= view(A,K,J)
            end
            # above, B but not A
            for K = Block.(max(1,J̃-B_u):min(N,J̃-A_u-1))
                view(C,K,J) .= $op.(view(B,K,J))
            end
            # below, A but not B
            for K = Block.(max(1,J̃+B_l+1):min(N,J̃+A_l))
                view(C,K,J) .= view(A,K,J)
            end
            # below, B but not A
            for K = Block.(max(1,J̃+A_l+1):min(N,J̃+B_l))
                view(C,K,J) .= $op.(view(B,K,J))
            end
            # above, neither
            for K = Block.(max(J̃-C_u,1):min(J̃-max(A_u,B_u)-1,N))
                view(C,K,J) .= zero(T)
            end
            # below, neither
            for K = Block.(max(J̃+max(A_l,B_l)+1,1):min(J̃+C_l,N))
                view(C,K,J) .= zero(T)
            end
        end

        C
    end
end

##
# special cases to hack tests where 2.0 .* A .+ B can be constructed in different ways depending on REPL or in a function.
# Should be made more general.
##

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
    BlockBandedMatrix{T}(undef, axes(bc), (max(Al,Bl), max(Au,Bu)))
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

    BandedBlockBandedMatrix{T}(undef, axes(bc), (max(Al,Bl), max(Au,Bu)), (max(Aλ,Bλ), max(Aμ,Bμ)))
end

