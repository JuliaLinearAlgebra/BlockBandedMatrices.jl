Base.transpose(bs::BlockBandedSizes) = BlockBandedSizes(
    BlockSizes(reverse(bs.block_sizes.cumul_sizes)),
    blockbandwidth(bs, 2), blockbandwidth(bs, 1))
Base.transpose(bs::BandedBlockBandedSizes) = BandedBlockBandedSizes(
    BlockSizes(reverse(bs.block_sizes.cumul_sizes)),
    bs.u, bs.l, bs.μ, bs.λ)
blocksizes(A::AdjOrTrans) = transpose(blocksizes(parent(A)))
blockbandwidths(A::AdjOrTrans) = reverse(blockbandwidths(parent(A)))
subblockbandwidths(A::AdjOrTrans) = reverse(subblockbandwidths(parent(A)))
