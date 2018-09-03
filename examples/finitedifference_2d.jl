###########
# This gives an example of Gauss–Seidel
#
###########

using BlockBandedMatrices, BandedMatrices


function finitedifference_2d(n)
    h = 1/n
    D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2
    D_xx = BandedBlockBandedMatrix(Kron(D², Eye(n)))
    D_yy = BandedBlockBandedMatrix(Kron(Eye(n), D²))
    D_xx + D_yy
end

function gaussseidel(L, U, b, x=copy(b), M=5)
    for _=1:M
        @view(x[1:end-1]) .= Mul(U , @view(x[2:end]))
        x[end] = 0
        x .= b .- x
        x .= Ldiv(L, x)
    end
    x
end

n = 10
Δt = (1/n^2)/4; Δ = finitedifference_2d(n); A = I - Δt*Δ  # 16k x 16k discretization

L = LowerTriangular(A)
U = UpperTriangular(@view A[1:end-1,2:end])

b = randn(size(A,1));
@time u = A\b; # 6s
x = copy(b)
@time gaussseidel(L,U, b, x, 20) # 1.6s
norm(x - u) # 6*10^(-10)
