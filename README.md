# BlockBandedMatrices.jl
A Julia package for representing block-block-banded matrices and banded-block-banded matrices

[![Build Status](https://travis-ci.org/JuliaMatrices/BlockBandedMatrices.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/BlockBandedMatrices.jl)

<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMatrices.github.io/BlockBandedMatrices.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMatrices.github.io/BlockBandedMatrices.jl/latest) -->



This package supports representing block-banded and banded-block-banded matrices by only
storing the entries in the non-zero bands.


A `BlockBandedMatrix` is a subtype of `BlockMatrix` of [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl)
whose non-zero blocks are banded. We can construct a `BlockBandedMatrix` as follows:
```julia
l,u = 2,1          # block bandwidths
N = M = 4          # number of row/column blocks
cols = rows = 1:N  # block sizes

BlockBandedMatrix(Zeros(sum(rows),sum(cols)), (rows,cols), (l,u)) # creates a block-banded matrix of zeros
BlockBandedMatrix(Zeros(sum(rows),sum(cols)), (rows,cols), (l,u)) # creates a block-banded matrix with ones in the non-zero entries
BlockBandedMatrix(I, (rows,cols), (l,u))                          # creates a block-banded  identity matrix
```

A `BandedBlockBandedMatrix` has the added structure that the blocks themselves are
banded, and conform to the banded matrix interface of [BandedMatrices.jl](https://github.com/JuliaMatrices/BandedMatrices.jl).
We can construct a `BandedBlockBandedMatrix` as follows:
```julia
l,u = 2,1          # block bandwidths
λ,μ = 1,2          # sub-block bandwidths: the bandwidths of each block
N = M = 4          # number of row/column blocks
cols = rows = 1:N  # block sizes
BandedBlockBandedMatrix(Zeros(sum(rows),sum(cols)), (rows,cols), (l,u), (λ,μ)) # creates a banded-block-banded matrix of zeros
BandedBlockBandedMatrix(Ones(sum(rows),sum(cols)), (rows,cols), (l,u), (λ,μ))  # creates a banded-block-banded matrix with ones in the non-zero entries
BandedBlockBandedMatrix(I, (rows,cols), (l,u), (λ,μ)))                         # creates a banded-block-banded identity matrix
```


## Implementation

A `BlockBandedMatrix` stores the entries in a single vector, ordered by columns.
For example, if `A` is a `BlockBandedMatrix` with block-bandwidths `(A.l,A.u) == (1,0)`
and the block sizes `fill(2, N)` where `N = 3` is the number
of row and column blocks, then `A` has zero structure
```julia
[ a_11 a_12
  a_21 a_22
  a_31 a_32 a_33 a_34
  a_41 a_42 a_43 a_44  
            a_53 a_54
            a_63 a_64 ]
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
[ a_11
  a_21 a_22
  a_31      a_33
  a_41 a_42 a_43 a_44  
            a_53    
            a_63 a_64 ]
```
and is stored in memory via `A.data` as a `PseudoBlockMatrix`, which has block sizes
2 x 2, containing entries:
```
[a_11 a_22 a_33 a_44;
 a_21 X    a_43 X   ;
 a_31 a_42 a_53 a_64;
 a_41 X    a_63 X ]
```
where `X` is an entry that is not used.

The reasoning behind this storage scheme as that each block still satisfies
the banded matrix interface. 
