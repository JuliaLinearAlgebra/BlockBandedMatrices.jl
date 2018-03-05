using BlockArrays, BlockBandedMatrices, Compat.Test
    import BlockBandedMatrices: BlockIndexRange, BlockSlice


B = Block(2)
Bi = B[2:3]

@test Block(Bi) == B
@test collect(Bi) == [BlockIndex((2,), 2), BlockIndex((2,), 3)]

A = PseudoBlockArray(rand(4), [1,3])

@test BlockBandedMatrices._unblock(A.block_sizes.cumul_sizes[1], (Bi,)) ==
        BlockArrays.unblock(A, indices(A), (Bi, )) == BlockSlice(Bi, 3:4) ==
        parentindexes(view(A, Bi))[1] == BlockSlice(Bi, 3:4)

@test A[Bi] == A[3:4]
