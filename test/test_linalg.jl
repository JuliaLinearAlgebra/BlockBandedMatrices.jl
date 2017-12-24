using BlockArrays, BlockBandedMatrices, Compat.Test
    import BlockBandedMatrices: _BandedBlockBandedMatrix, scalemul!, _scalemul!, memorylayout


l , u = 1,1
λ , μ = 1,1
N = M = 10
cols = rows = 1:N

data = reshape(Vector{Float64}(1:(λ+μ+1)*(l+u+1)*sum(cols)), ((λ+μ+1)*(l+u+1), sum(cols)))
A = _BandedBlockBandedMatrix(data, (rows,cols), (l,u), (λ,μ))

V = view(A, Block(2), Block(2))
@test unsafe_load(Base.unsafe_convert(Ptr{Float64}, V)) == 13.0




@which Base.unsafe_convert(Ptr{Float64}, V)

C =  A*A
@test C isa BandedBlockBandedMatrix
@test Matrix(A*A) ≈ Matrix(A)*Matrix(A)
@test C.l == C.u == C.λ == C.μ == 2
