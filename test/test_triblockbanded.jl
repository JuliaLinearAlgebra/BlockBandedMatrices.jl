using BlockBandedMatrices, LazyArrays
import BlockBandedMatrices: MemoryLayout, UpperTriangularLayout, BandedBlockBandedColumnMajor,
                        BandedColumnMajor
A = BandedBlockBandedMatrix{Float64}(undef, (1:10,1:10), (1,1), (1,1))
    A.data .= randn.()
    A
U = UpperTriangular(A)


@test MemoryLayout(U) == UpperTriangularLayout(BandedBlockBandedColumnMajor())

@test MemoryLayout(view(U, Block(1,1))) == UpperTriangularLayout(BandedColumnMajor())
@test MemoryLayout(view(U, Block(1,2))) == BandedColumnMajor()


@test Base.return_types(MemoryLayout, (typeof(view(U, Block(1,2))),))[1] ==
            Union{BandedColumnMajor, UpperTriangularLayout{BandedColumnMajor}}
