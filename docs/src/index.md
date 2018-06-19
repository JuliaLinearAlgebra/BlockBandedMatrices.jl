# BandedMatrices.jl Documentation


## Creating block-banded and banded-block-banded matrices

```@docs
BlockBandedMatrix
```

```@docs
BandedBlockBandedMatrix
```


## Accessing block-banded and banded-block-banded matrices

```@docs
blockbandwidths
```

```@docs
blockbandwidth
```

```@docs
subblockbandwidths
```

```@docs
subblockbandwidth
```

## Implementation

A `BlockBandedMatrix` stores the entries in a single vector, ordered by columns.
For example, if `A` is a `BlockBandedMatrix` with block-bandwidths `(A.l,A.u) == (1,0)`
and the block sizes `fill(2, N)` where `N = 3` is the number
of row and column blocks, then `A` has zero structure
```julia
[ a_11 a_12 │  ⋅    ⋅
  a_21 a_22 │  ⋅    ⋅
  ──────────┼──────────
  a_31 a_32 │ a_33 a_34
  a_41 a_42 │ a_43 a_44  
  ──────────┼──────────
   ⋅    ⋅   │ a_53 a_54
   ⋅    ⋅   │ a_63 a_64 ]
```
and is stored in memory via `A.data` as a single vector by columns, containing:
```
[a_11,a_21,a_31,a_41,a_12,a_22,a_32,a_42,a_33,a_43,a_53,a_63,a_34,a_44,a_54,a_64]
```
The reasoning behind this storage scheme as that each block still satisfies
the strided matrix interface, but we can also use BLAS and LAPACK to, for example,
upper-triangularize a block column all at once.


A `BandedBlockBandedMatrix` stores the entries as a `PseudoBlockMatrix`,
with the number of row blocks equal to `A.l + A.u + 1`, and the row
block sizes are all `A.μ + A.λ + 1`. The column block sizes of the storage is
the same as the the column block sizes of the `BandedBlockBandedMatrix`. This
is a block-wise version of the storage of `BandedMatrix`.

For example, if `A` is a `BandedBlockBandedMatrix` with block-bandwidths `(A.l,A.u) == (1,0)`
and subblock-bandwidths `(A.λ, A.μ) == (1,0)`, and the block sizes `fill(2, N)` where `N = 3` is the number
of row and column blocks, then `A` has zero structure
```julia
[ a_11  ⋅   │  ⋅    ⋅
  a_21 a_22 │  ⋅    ⋅
  ──────────┼──────────
  a_31  ⋅   │ a_33  ⋅
  a_41 a_42 │ a_43 a_44  
  ──────────┼──────────
   ⋅    ⋅   │ a_53  ⋅
   ⋅    ⋅   │ a_63 a_64 ]
```
and is stored in memory via `A.data` as a `PseudoBlockMatrix`, which has block sizes
2 x 2, containing entries:
```julia
[a_11 a_22 │ a_33 a_44
 a_21  ×   │ a_43  ×  
 ──────────┼──────────
 a_31 a_42 │ a_53 a_64
 a_41  ×   │ a_63  ×   ]
```
where `×` is an entry in memory that is not used.

The reasoning behind this storage scheme as that each block still satisfies
the banded matrix interface.
