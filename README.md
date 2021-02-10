# BlockBandedMatrices.jl
A Julia package for representing block-banded matrices and banded-block-banded matrices

[![Build Status](https://github.com/JuliaMatrices/BlockBandedMatrices.jl/workflows/CI/badge.svg)](https://github.com/JuliaMatrices/BlockBandedMatrices.jl/actions)
[![codecov](https://codecov.io/gh/JuliaMatrices/BlockBandedMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMatrices/BlockBandedMatrices.jl)


 [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMatrices.github.io/BlockBandedMatrices.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMatrices.github.io/BlockBandedMatrices.jl/latest)




This package supports representing block-banded and banded-block-banded matrices by only
storing the entries in the non-zero bands.


A `BlockBandedMatrix` is a subtype of `BlockMatrix` of [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl)
whose layout of non-zero blocks is banded. We can construct a `BlockBandedMatrix` as follows:
```julia
l,u = 2,1          # block bandwidths
N = M = 4          # number of row/column blocks
cols = rows = 1:N  # block sizes

BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u)) # creates a block-banded matrix of zeros
BlockBandedMatrix(Ones(sum(rows),sum(cols)), rows,cols, (l,u)) # creates a block-banded matrix with ones in the non-zero entries
BlockBandedMatrix(I, rows,cols, (l,u))                          # creates a block-banded  identity matrix
```

A `BandedBlockBandedMatrix` has the added structure that the blocks themselves are
banded, and conform to the banded matrix interface of [BandedMatrices.jl](https://github.com/JuliaMatrices/BandedMatrices.jl).
We can construct a `BandedBlockBandedMatrix` as follows:
```julia
l,u = 2,1          # block bandwidths
λ,μ = 1,2          # sub-block bandwidths: the bandwidths of each block
N = M = 4          # number of row/column blocks
cols = rows = 1:N  # block sizes
BandedBlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u), (λ,μ)) # creates a banded-block-banded matrix of zeros
BandedBlockBandedMatrix(Ones(sum(rows),sum(cols)), rows,cols, (l,u), (λ,μ))  # creates a banded-block-banded matrix with ones in the non-zero entries
BandedBlockBandedMatrix(I, rows,cols, (l,u), (λ,μ))                          # creates a banded-block-banded identity matrix
```
