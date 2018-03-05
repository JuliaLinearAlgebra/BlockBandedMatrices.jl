struct BlockIndexRange{N,R<:NTuple{N,AbstractUnitRange{Int}}}
    block::Block{N,Int}
    indices::R
end

"""
    BlockIndexRange(block, startind:stopind)

represents a cartesian range inside a  block.
"""
BlockIndexRange

BlockIndexRange(block::Block{N}, inds::NTuple{N,AbstractUnitRange{Int}}) where {N} =
    BlockIndexRange{N,typeof(inds)}(inds)
BlockIndexRange(block::Block{N}, inds::Vararg{AbstractUnitRange{Int},N}) where {N} =
    BlockIndexRange(block,inds)

getindex(B::Block{N}, inds::Vararg{AbstractUnitRange{Int},N}) where N = BlockIndexRange(B,inds)

eltype(R::BlockIndexRange) = eltype(typeof(R))
eltype(::Type{BlockIndexRange{N}}) where {N} = BlockIndex{N}
eltype(::Type{BlockIndexRange{N,R}}) where {N,R} = BlockIndex{N}
if VERSION < v"0.7.0-DEV.4043"
    iteratorsize(::Type{<:BlockIndexRange}) = Base.HasShape()
else
    IteratorSize(::Type{<:BlockIndexRange}) = Base.HasShape{1}()
end

first(iter::BlockIndexRange) = BlockIndex(iter.block.n, map(first, iter.indices))
last(iter::BlockIndexRange)  = BlockIndex(iter.block.n, map(last, iter.indices))


@inline function start(iter::BlockIndexRange)
    iterfirst, iterlast = first(iter), last(iter)
    if any(map(>, iterfirst.α, iterlast.α))
        return BlockIndex(iterlast.I, iterlast.α .+ 1)
    end
    iterfirst
end
@inline function next(iter::BlockIndexRange, state)
    state, BlockIndex(state.I, inc(state.α, first(iter).α, last(iter).α))
end
# increment & carry
@inline inc(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline inc(state::Tuple{Int}, start::Tuple{Int}, stop::Tuple{Int}) = (state[1]+1,)
@inline function inc(state, start, stop)
    if state[1] < stop[1]
        return (state[1]+1,tail(state)...)
    end
    newtail = inc(tail(state), tail(start), tail(stop))
    (start[1], newtail...)
end
@inline done(iter::BlockIndexRange, state) = state.α[end] > last(iter.indices[end])

size(iter::BlockIndexRange) = map(dimlength, first(iter).α, last(iter).α)
dimlength(start, stop) = stop-start+1

length(iter::BlockIndexRange) = prod(size(iter))


Block(bs::BlockIndexRange) = bs.block


### Views
Block(bs::BlockSlice{<:BlockIndexRange}) = Block(bs.block)




function _unblock(cum_sizes, I::Tuple{BlockIndexRange{1,R}, Vararg{Any}}) where {R}
    B = Block(first(I))
    range = cum_sizes[Int(B)]-1 + first(I).indices[1]

    BlockSlice(I[1], range)
end


to_index(::BlockIndexRange) = throw(ArgumentError("BlockIndexRange must be converted by to_indices(...)"))

@inline to_indices(A, inds, I::Tuple{BlockIndexRange{1,R}, Vararg{Any}}) where R =
    (unblock(A, inds, I), to_indices(A, _maybetail(inds), tail(I))...)

# splat out higher dimensional blocks
# this mimics view of a CartesianIndex
@inline to_indices(A, inds, I::Tuple{BlockIndexRange, Vararg{Any}}) =
    to_indices(A, inds, (BlockRange.(Block.(I[1].block.n), tuple.(I[1].indices))..., tail(I)...))


# In 0.7, we need to override to_indices to avoid calling linearindices
@inline to_indices(A, I::Tuple{BlockIndexRange, Vararg{Any}}) =
    to_indices(A, axes(A), I)


# #################
# # support for pointers
# #################
#
# function unsafe_convert(::Type{Ptr{T}},
#                         V::SubArray{T, N, BlockArray{T, N, AT}, NTuple{N, BlockSlice{Block{1,Int}}}}) where AT <: AbstractArray{T, N} where {T,N}
#     unsafe_convert(Ptr{T}, parent(V).blocks[Int.(Block.(parentindices(V)))...])
# end
