
blockbandwidths(A::AdjOrTrans) = reverse(blockbandwidths(parent(A)))
subblockbandwidths(A::AdjOrTrans) = reverse(subblockbandwidths(parent(A)))
