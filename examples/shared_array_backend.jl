using BlockArrays: _BlockArray, PseudoBlockArray, BlockArray, BlockMatrix, BlockVector,
                  nblocks, Block, cumulsizes, AbstractBlockVector
using BlockBandedMatrices: BandedBlockBandedMatrix, _BandedBlockBandedMatrix,
                           blockbandwidths, subblockbandwidths, blockbandwidth,
                           BandedBlockBandedSizes
using LinearAlgebra: BLAS
import LinearAlgebra
using BandedMatrices: _BandedMatrix, BandedMatrix
using SharedArrays
using LazyArrays
using Distributed: procs, remotecall_wait
import Distributed

import Adapt: adapt

adapt(T::Type, b::BandedBlockBandedMatrix) =
    _BandedBlockBandedMatrix(adapt(T, b.data), b.block_sizes)
adapt(T::Type{<:AbstractArray}, b::PseudoBlockArray) =
    PseudoBlockArray(T(b.blocks), b.block_sizes)


const SharedBandedBlockBandedMatrix =
    BandedBlockBandedMatrix{T, PseudoBlockArray{T, 2, SharedArray{T, 2}}} where T

function SharedBandedBlockBandedMatrix{T}(::UndefInitializer,
                                          bs::BandedBlockBandedSizes;
                                          kwargs...) where T
  Block = fieldtype(SharedBandedBlockBandedMatrix{T}, :data)
  Shared = fieldtype(Block, :blocks)
  kwargs = Dict(kwargs)
  init = pop!(kwargs, :init, nothing)
  shared = Shared(size(bs); kwargs...)
  result = _BandedBlockBandedMatrix(Block(shared, bs.data_block_sizes), bs)
  populate!(result, init)
  result
end

Distributed.procs(A::SharedBandedBlockBandedMatrix) = procs(A.data.blocks)

function populate!(A::SharedBandedBlockBandedMatrix, range, block_populate!::Function)
  k = 1
  for i in 1:nblocks(A, 1), j in max(i - A.u, 1):min(i + A.l, nblocks(A, 2))
    if k in range
      block_populate!(view(A, Block(i, j)), i, j)
    end
    k += 1
  end
  A
end


function populate!(A::SharedBandedBlockBandedMatrix, block_populate!::Function)
  n = nnzero(nblocks(A)..., A.l, A.u)
  m = length(procs(A))
  @sync begin
    for (i, proc) in enumerate(procs(A))
      start = (n ÷ m) * (i - 1) + min((n % m), i - 1) + 1
      stop = (n ÷ m) * i + min((n % m), i)
      @async remotecall_wait(populate!, proc, A, start:stop, block_populate!)
    end
  end
  A
end

populate!(block_populate!::Function, A::SharedBandedBlockBandedMatrix) =
    populate!(A, block_populate!)

SharedBandedBlockBandedMatrix{T}(init::Function,
                                 bs::BandedBlockBandedSizes;
                                 pids=Int[]) where T =
    SharedBandedBlockBandedMatrix{T}(undef, bs; pids=pids, init=init)
function SharedBandedBlockBandedMatrix{T}(init::Function,
                                          dims::NTuple{2, AbstractVector{Int}},
                                          lu::NTuple{2, Int}, λμ::NTuple{2, Int};
                                          pids=Int[]) where T
  bs = BandedBlockBandedSizes(dims..., lu..., λμ...)
  SharedBandedBlockBandedMatrix{T}(init, bs; pids=pids)
end

"""Number of non-zero elements in an banded matrix"""
function nnzero(n::Integer, m::Integer, l::Integer, u::Integer)
  result = zero(n)
  for i = 0:min(n, l)
    result += min(n - i, m)
  end
  for i = 1:min(m, u)
    result += min(m - i, n)
  end
  result
end

function LinearAlgebra.mul!(c::AbstractBlockVector{T},
                            A::SharedBandedBlockBandedMatrix{T},
                            x::AbstractBlockVector{T}) where T
    @assert nblocks(A, 1) == nblocks(c, 1)
    @assert cumulsizes(A, 1) == cumulsizes(c, 1)
    @assert nblocks(A, 2) == nblocks(x, 1)
    @assert cumulsizes(A, 2) == cumulsizes(x, 1)

    n = nblocks(A, 1)
    m = length(procs(A))

    @sync for (p, proc) in enumerate(procs(A))

        p > n && continue
        start = (n ÷ m) * (p - 1) + min((n % m), p - 1) + 1
        stop = (n ÷ m) * p + min((n % m), p)

        @async begin
          remotecall_wait(proc, start:stop) do irange
            @inbounds for i in irange
                fill!(view(c, Block(i)), zero(eltype(c)))
                for j = max(1, i - A.l):min(nblocks(A, 2), i + A.u)
                    c[Block(i)] .+= Mul(view(A, Block(i, j)), view(x, Block(j)))
                end
            end
          end
        end

    end
    c
end


using Test

function testme()
  SBBB = SharedBandedBlockBandedMatrix
  @testset "shared array backend" begin

    @testset "Initialization" begin
      n, m = repeat([2], 4), repeat([3], 2)
      A = SBBB{Int64}((n, m), (1, 1), (1, 0)) do block, i, j
        block .= 0
        if (i == 3) && (j == 2); block[2, 2] = 1 end
      end
      @test view(A, Block(3, 2))[2, 2] == 1
      view(A, Block(3, 2))[2, 2] = 0
      @test all(A .== 0)
    end

    @testset "count non-zero elements" begin
        for i in 1:100
          n, m = rand(1:10, 2)
          l, u = rand(0:10, 2)
          A = BandedMatrix{Int8}(undef, n, m, l, u)
          A.data .= 1
          @test sum(A) == nnzero(n, m, l, u)
        end
    end

    @testset "Multiplication" begin
       N, M = rand(1:3, 2)
       l, u, λ, μ = rand(0:2, 4)
       n, m = rand(max(l, u, λ, μ):20, N), rand(max(l, u, λ, μ):20, M)
       A = BandedBlockBandedMatrix{Float64}(undef, (n, m), (l, u), (λ, μ))
       A.data .= rand.()
       x = PseudoBlockArray(Array{Float64, 1}(undef, size(A, 2)), m)
       x .= rand.()

       Ashared = adapt(SharedArray, A)
       @test Ashared.data.blocks isa SharedArray
       @test Ashared isa SharedBandedBlockBandedMatrix
       @test length(procs(Ashared)) == max(1, length(procs()) - 1)
       cshared = adapt(SharedArray,
                      PseudoBlockArray(Array{Float64, 1}(undef, size(A, 1)), n))
       @test cshared.blocks isa SharedArray
       cshared .= rand.()
       xshared = adapt(SharedArray, x)

       @test LinearAlgebra.mul!(cshared, Ashared, xshared) ≈ A * x
    end

  end
end
