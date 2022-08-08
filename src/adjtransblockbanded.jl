
blockbandwidths(A::AdjOrTrans) = reverse(blockbandwidths(parent(A)))
subblockbandwidths(A::AdjOrTrans) = reverse(subblockbandwidths(parent(A)))
transposelayout(::AbstractBlockBandedLayout) = BlockBandedLayout()
transposelayout(::AbstractBandedBlockBandedLayout) = BandedBlockBandedLayout()
conjlayout(::Type{<:Complex}, ::M) where M<:AbstractBandedBlockBandedLayout = ConjLayout{M}()
conjlayout(::Type{<:Complex}, ::M) where M<:AbstractBlockBandedLayout = ConjLayout{M}()
