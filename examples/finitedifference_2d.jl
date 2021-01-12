###########
# This gives an example of Gauss–Seidel
#
###########

using BlockBandedMatrices, BandedMatrices, ArrayLayouts, FillArrays, LazyBandedMatrices, LazyArrays


function finitedifference_2d(n)
    h = 1/n
    D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))/h^2
    D_xx = BandedBlockBandedMatrix(Kron(D², Eye(n)))
    D_yy = BandedBlockBandedMatrix(Kron(Eye(n), D²))
    D_xx + D_yy
end

function _gaussseidel(L, U, b, x=copy(b), y=copy(b), M=5)
    for _=1:M
        @view(y[1:end-1]) .= Mul(U , @view(x[2:end]))
        y[end] = 0
        x .= b .- y
        x .= Ldiv(L, x)
    end
    x
end



function _gaussseidel2(L, U, b, x=copy(b), y=copy(b), M=5)
    for _=1:M
        mul!(@view(y[1:end-1]) , U , @view(x[2:end]))
        y[end] = 0
        y .= b .- y
        x .= L\ y
    end
    x
end

function gaussseidel(A, b, M)
    n = Int(sqrt(length(b)))
    L = LowerTriangular(A)
    U = BandedBlockBandedMatrix(UpperTriangular(@view(A[1:end-1,2:end])), ([fill(n,n-1); n-1], [n-1; fill(n,n-1)]),
                                            (0,1), (0,1))
    x = copy(b)
    _gaussseidel(L,U, b, x, copy(x), M) # 1.6s
    x
end

n = 1000;
    Δt = (1/n^2)/4; @time Δ = finitedifference_2d(n);
    @time A = I - Δt*Δ  # 1m x 1m discretization
    b = randn(n^2)
    L = LowerTriangular(A)
    @time U = BandedBlockBandedMatrix(UpperTriangular(@view(A[1:end-1,2:end])), ([fill(n,n-1); n-1], [n-1; fill(n,n-1)]),
                                            (0,1), (0,1))
    x = copy(b)
    y = copy(x)
    @time _gaussseidel(L,U, b, x, y, 20) # 0.4s

    @time L̃ = sparse(BandedBlockBandedMatrix(L))
    @time Ũ = sparse(U)
    x = copy(b)
    y = copy(x)
    @time _gaussseidel2(L̃,Ũ, b, x, y, 20)

Ã = sparse(A)
@time qr(Ã);
@time Ã \b ;

A = randn(n,n)

n = 1000;
    As = [randn(2k+1,k) for k=1:n];

    @time for A in As
        qr(A)
    end
n = 1000;
    # As = [randn(2k+1,k) for k=1:n];

    @time for A in As
        lu(A)
    end

