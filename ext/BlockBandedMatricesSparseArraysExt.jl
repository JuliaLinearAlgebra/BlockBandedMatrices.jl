module BlockBandedMatricesSparseArraysExt

using BlockBandedMatrices
using BandedMatrices
using BandedMatrices: _banded_rowval, _banded_colval, _banded_nzval
using BlockArrays: blockaxes, blockcolsupport
import SparseArrays: sparse

function sparse(A::BandedBlockBandedMatrix)
    i = Vector{Int}()
    j = Vector{Int}()
    z = Vector{eltype(A)}()
    for J = blockaxes(A,2), K = blockcolsupport(A, J)
        B = view(A, K, J)
        ĩ = _banded_rowval(B)
        j̃ = _banded_colval(B)
        z̃ = _banded_nzval(B)
        ĩ .+= first(axes(A,1)[K])-1
        j̃ .+= first(axes(A,2)[J])-1
        append!(i, ĩ)
        append!(j, j̃)
        append!(z, z̃)
    end
    sparse(i, j, z, size(A)...)
end

end
