
# function blockmatrix_αA_mul_B_plus_βC!(α,A,x,β,y)
#     if length(x) != size(A,2) || length(y) != size(A,1)
#         throw(BoundsError())
#     end
#
#     BLAS.scal!(length(y),β,y,1)
#     o=one(eltype(y))
#
#     for J=Block(1):Block(blocksize(A,2))
#         jr=blockcols(A,J)
#         for K=blockcolrange(A,J)
#             kr=blockrows(A,K)
#             B=view(A,K,J)
#             αA_mul_B_plus_βC!(α,B,view(x,jr),o,view(y,kr))
#         end
#     end
#     y
# end
#
# αA_mul_B_plus_βC!(α,A::AbstractBlockMatrix,x::AbstractVector,β,y::AbstractVector) =
#     blockmatrix_αA_mul_B_plus_βC!(α,A,x,β,y)
#
# Base.A_mul_B!(y::Vector,A::AbstractBlockMatrix,b::Vector) =
#     αA_mul_B_plus_βC!(one(eltype(A)),A,b,zero(eltype(y)),y)
#
#
#
#
# function block_axpy!(α,A,Y)
#     if size(A) ≠ size(Y)
#         throw(BoundsError())
#     end
#
#     for J=Block.(1:blocksize(A,2)), K=blockcolrange(A,J)
#         BLAS.axpy!(α,view(A,K,J),view(Y,K,J))
#     end
#     Y
# end
#
# Base.BLAS.axpy!(α,A::AbstractBlockMatrix,Y::AbstractBlockMatrix) = block_axpy!(α,A,Y)
#
#
#
# ## Algebra
#
#
# function Base.A_mul_B!(Y::AbstractBlockMatrix,A::AbstractBlockMatrix,B::AbstractBlockMatrix)
#     T=eltype(Y)
#     BLAS.scal!(length(Y.data),zero(T),Y.data,1)
#     o=one(T)
#     for J=Block(1):Block(blocksize(B,2)),N=blockcolrange(B,J),K=blockcolrange(A,N)
#         αA_mul_B_plus_βC!(o,view(A,K,N),view(B,N,J),o,view(Y,K,J))
#     end
#     Y
# end


######
# back substitution
######
# #TODO: don't auto-pad
# function trtrs!(::Type{Val{'U'}}, A::BlockBandedMatrix, u::Vector)
#     # When blocks are square, use LAPACK trtrs!
#     mn=min(length(A.rows),length(A.cols))
#     if A.rows[1:mn] == A.cols[1:mn]
#         blockbanded_squareblocks_trtrs!(A,u)
#     else
#         blockbanded_rectblocks_trtrs!(A,u)
#     end
# end
#
#
#
# function blockbanded_squareblocks_trtrs!(A::BlockBandedMatrix,u::Vector)
#     if size(A,1) < size(u,1)
#         throw(BoundsError())
#     end
#     n=size(u,1)
#     N=Block(A.rowblocks[n])
#
#     kr1=blockrows(A,N)
#     b=n-kr1[1]+1
#     kr1=kr1[1]:n
#
#     trtrs!('U','N','N',view(A,N[1:b],N[1:b]),view(u,kr1))
#
#     for K=N-1:-1:Block(1)
#         kr=blockrows(A,K)
#         for J=min(N,blockrowstop(A,K)):-1:K+1
#             if J==N  # need to take into account zeros
#                 gemv!('N',-one(eltype(A)),view(A,K,N[1:b]),view(u,kr1),one(eltype(A)),view(u,kr))
#             else
#                 gemv!('N',-one(eltype(A)),view(A,K,J),view(u,blockcols(A,J)),one(eltype(A)),view(u,kr))
#             end
#         end
#         trtrs!('U','N','N',view(A,K,K),view(u,kr))
#     end
#
#     u
# end
#
# function blockbanded_rectblocks_trtrs!(R::BlockBandedMatrix{T},b::Vector) where T
#     n=n_end=length(b)
#     K_diag=N=Block(R.rowblocks[n])
#     J_diag=M=Block(R.colblocks[n])
#
#     while n > 0
#         B_diag = view(R,K_diag,J_diag)
#
#         kr = blockrows(R,K_diag)
#         jr = blockcols(R,J_diag)
#
#
#         k = n-kr[1]+1
#         j = n-jr[1]+1
#
#         skr = max(1,k-j+1):k   # range in the sub block
#         sjr = max(1,j-k+1):j   # range in the sub block
#
#         kr2 = kr[skr]  # diagonal rows/cols we are working with
#
#         for J = min(M,blockrowstop(R,K_diag)):-1:J_diag+1
#             B=view(R,K_diag,J)
#             Sjr = blockcols(R,J)
#
#             if J==M
#                 Sjr = Sjr[1]:n_end  # The sub rows of the rhs we will multiply
#                 gemv!('N',-one(T),view(B,skr,1:length(Sjr)),
#                                     view(b,Sjr),one(T),view(b,kr2))
#             else  # can use all columns
#                 gemv!('N',-one(T),view(B,skr,:),
#                                     view(b,Sjr),one(T),view(b,kr2))
#             end
#         end
#
#         if J_diag ≠ M && sjr[end] ≠ size(B_diag,2)
#             # subtract non-triangular columns
#             sjr2 = sjr[end]+1:size(B_diag,2)
#             gemv!('N',-one(T),view(B_diag,skr,sjr2),
#                             view(b,sjr2 + jr[1]-1),one(T),view(b,kr2))
#         elseif J_diag == M && sjr[end] ≠ size(B_diag,2)
#             # subtract non-triangular columns
#             Sjr = jr[1]+sjr[end]:n_end
#             gemv!('N',-one(T),view(B_diag,skr,sjr[end]+1:sjr[end]+length(Sjr)),
#                             view(b,Sjr),one(T),view(b,kr2))
#         end
#
#         trtrs!('U','N','N',view(B_diag,skr,sjr),view(b,kr2))
#
#         if k == j
#             K_diag -= 1
#             J_diag -= 1
#         elseif j < k
#             J_diag -= 1
#         else # if k < j
#             K_diag -= 1
#         end
#
#         n = kr2[1]-1
#     end
#     b
# end
#
#
# function trtrs!(A::BlockBandedMatrix{T},u::Matrix) where T
#     if size(A,1) < size(u,1)
#         throw(BoundsError())
#     end
#     n=size(u,1)
#     N=Block(A.rowblocks[n])
#
#     kr1=blockrows(A,N)
#     b=n-kr1[1]+1
#     kr1=kr1[1]:n
#
#     trtrs!('U','N','N',view(A,N[1:b],N[1:b]),view(u,kr1,:))
#
#     for K=N-1:-1:Block(1)
#         kr=blockrows(A,K)
#         for J=min(N,blockrowstop(A,K)):-1:K+1
#             if J==N  # need to take into account zeros
#                 gemm!('N',-one(T),view(A,K,N[1:b]),view(u,kr1,:),one(T),view(u,kr,:))
#             else
#                 gemm!('N',-one(T),view(A,K,J),view(u,blockcols(A,J),:),one(T),view(u,kr,:))
#             end
#         end
#         trtrs!('U','N','N',view(A,K,K),view(u,kr,:))
#     end
#
#     u
# end
