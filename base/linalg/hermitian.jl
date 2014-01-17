## Hermitian matrices

immutable Hermitian{T<:Number} <: AbstractMatrix{T}
    S::Matrix{T}
    uplo::Char
end
function Hermitian(S::Matrix, uplo::Symbol)
    chksquare(S)
    Hermitian(S, string(uplo)[1])
end
Hermitian(A::Matrix) = Hermitian(A, :U)

convert{T1,T2}(::Type{Hermitian{T1}},A::Hermitian{T2}) = Hermitian(convert(Matrix{T1},A.S),A.uplo)
copy(A::Hermitian) = Hermitian(copy(A.S),A.uplo)
size(A::Hermitian, args...) = size(A.S, args...)
getindex(A::Hermitian, i::Integer, j::Integer) = (A.uplo == 'U') == (i < j) ? getindex(A.S, i, j) : conj(getindex(A.S, j, i))
full(A::Hermitian) = copytri!(A.S, A.uplo, true)
ishermitian(A::Hermitian) = true
issym{T<:Real}(A::Hermitian{T}) = true
issym{T<:Complex}(A::Hermitian{T}) = all(imag(A.S) .== 0)
ctranspose(A::Hermitian) = A
similar(A::Hermitian, args...) = Hermitian(similar(A.S, args...), A.uplo)

*(A::Hermitian, B::Hermitian) = *(full(A), full(B))
*(A::Hermitian, B::StridedMatrix) = *(full(A), B)
*(A::StridedMatrix, B::Hermitian) = *(A, full(B))

factorize(A::Hermitian) = bkfact(A.S, symbol(A.uplo), issym(A))
\(A::Hermitian, B::StridedVecOrMat) = \(bkfact(A.S, symbol(A.uplo), issym(A)), B)

eigfact!{T<:BlasFloat}(A::Hermitian{T}) = Eigen(LAPACK.syevr!('V', 'A', A.uplo, A.S, 0.0, 0.0, 0, 0, -1.0)...)
eigfact{T<:BlasFloat}(A::Hermitian{T}) = eigfact!(copy(A))
eigfact(A::Hermitian) = eigfact!(convert(Hermitian{promote_type(Float32,eltype(A))}, A))
eigvals!{T<:BlasFloat}(A::Hermitian{T}, il::Int, ih::Int) = LAPACK.syevr!('N', 'I', A.uplo, A.S, 0.0, 0.0, il, ih, -1.0)[1]
eigvals!{T<:BlasFloat}(A::Hermitian{T}, vl::Real, vh::Real) = LAPACK.syevr!('N', 'V', A.uplo, A.S, vl, vh, 0, 0, -1.0)[1]
# eigvals!(A::Hermitian, args...) = eigvals!(float(A), args...)
eigvals!(A::Hermitian) = eigvals!(A, 1, size(A, 1))
eigmax(A::Hermitian) = eigvals(A, size(A, 1), size(A, 1))[1]
eigmin(A::Hermitian) = eigvals(A, 1, 1)[1]

function eigfact!{T<:BlasFloat}(A::Hermitian{T}, B::Hermitian{T})
    vals, vecs, _ = LAPACK.sygvd!(1, 'V', A.uplo, A.S, B.uplo == A.uplo ? B.S : B.S')
    GeneralizedEigen(vals, vecs)
end
eigfact{T<:BlasFloat}(A::Hermitian{T}, B::Hermitian{T}) = eigfact!(copy(A), copy(B))
eigfact{TA,TB}(A::Hermitian{TA},B::Hermitian{TB}) = eigfact!(convert(Hermitian{promote_type(Float32,TA,TB)}, A), convert(Hermitian{promote_type(Float32,TA,TB)}, B))
eigvals!{T<:BlasFloat}(A::Hermitian{T}, B::Hermitian{T}) = LAPACK.sygvd!(1, 'N', A.uplo, A.S, B.uplo == A.uplo ? B.S : B.S')[1]

function expm(A::Hermitian)
    F = eigfact(A)
    scale(F[:vectors], exp(F[:values])) * F[:vectors]'
end

function sqrtm(A::Hermitian)
    F = eigfact(A)
    length(F[:values]) == 0 && return A
    vsqrt = sqrt(complex(F[:values]))
    all(imag(vsqrt) .== 0) && return F[:vectors]*Diagonal(real(vsqrt))*F[:vectors]'
    zc = complex(F[:vectors])
    return zc*Diagonal(vsqrt)*zc'
end
