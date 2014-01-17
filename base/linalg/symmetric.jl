# Symmetric matrices

immutable Symmetric{T<:Number} <: AbstractMatrix{T}
    S::Matrix{T}
    uplo::Char
end
function Symmetric{T<:Number}(S::Matrix{T}, uplo::Symbol)
    chksquare(S)
    Symmetric(S, string(uplo)[1])
end
Symmetric(A::StridedMatrix) = Symmetric(A, :U)

convert{T1,T2}(::Type{Symmetric{T1}},A::Symmetric{T2}) = Symmetric(convert(Matrix{T1},A.S),A.uplo)
copy(A::Symmetric) = Symmetric(copy(A.S),A.uplo)
size(A::Symmetric, args...) = size(A.S, args...)
getindex(A::Symmetric, i::Integer, j::Integer) = (A.uplo == 'U') == (i < j) ? getindex(A.S, i, j) : getindex(A.S, j, i)
full(A::Symmetric) = copytri!(A.S, A.uplo)
ishermitian{T<:Real}(A::Symmetric{T}) = true
ishermitian{T<:Complex}(A::Symmetric{T}) = all(imag(A.S) .== 0)
issym(A::Symmetric) = true
transpose(A::Symmetric) = A
similar(A::Symmetric, args...) = Symmetric(similar(A.S, args...), A.uplo)

*(A::Symmetric, B::Symmetric) = *(full(A), full(B))
*(A::Symmetric, B::StridedMatrix) = *(full(A), B)
*(A::StridedMatrix, B::Symmetric) = *(A, full(B))

factorize{T<:Real}(A::Symmetric{T}) = bkfact(A.S, symbol(A.uplo))
factorize{T<:Complex}(A::Symmetric{T}) = bkfact(A.S, symbol(A.uplo), true)
\(A::Symmetric, B::StridedVecOrMat) = \(bkfact(A.S, symbol(A.uplo), true), B)

eigfact!{T<:BlasReal}(A::Symmetric{T}) = Eigen(LAPACK.syevr!('V', 'A', A.uplo, A.S, 0.0, 0.0, 0, 0, -1.0)...)
eigfact{T<:BlasFloat}(A::Symmetric{T}) = eigfact!(copy(A))
eigfact(A::Symmetric) = eigfact!(convert(Symmetric{promote_type(Float32,eltype(A))},A))
eigvals!{T<:BlasReal}(A::Symmetric{T}, il::Int, ih::Int) = LAPACK.syevr!('N', 'I', A.uplo, A.S, 0.0, 0.0, il, ih, -1.0)[1]
eigvals!{T<:BlasReal}(A::Symmetric{T}, vl::Real, vh::Real) = LAPACK.syevr!('N', 'V', A.uplo, A.S, vl, vh, 0, 0, -1.0)[1]
# eigvals!(A::Symmetric, args...) = eigvals!(float(A), args...)
eigvals!(A::Symmetric) = eigvals!(A, 1, size(A, 1))
eigmax(A::Symmetric) = eigvals(A, size(A, 1), size(A, 1))[1]
eigmin(A::Symmetric) = eigvals(A, 1, 1)[1]

function eigfact!{T<:BlasReal}(A::Symmetric{T}, B::Symmetric{T})
    vals, vecs, _ = LAPACK.sygvd!(1, 'V', A.uplo, A.S, B.uplo == A.uplo ? B.S : B.S')
    GeneralizedEigen(vals, vecs)
end
eigfact{T<:BlasFloat}(A::Symmetric{T}, B::Symmetric{T}) = eigfact!(copy(A), copy(B))
eigfact{TA,TB}(A::Symmetric{TA}, B::Symmetric{TB}) = eigfact!(convert(Symmetric{promote_type(Float32, TA, TB)}, A), convert(Symmetric{promote_type(Float32, TA, TB)}, B))
eigvals!{T<:BlasReal}(A::Symmetric{T}, B::Symmetric{T}) = LAPACK.sygvd!(1, 'N', A.uplo, A.S, B.uplo == A.uplo ? B.S : B.S')[1]

function expm{T<:Real}(A::Symmetric{T})
    F = eigfact(A)
    scale(F[:vectors], exp(F[:values])) * F[:vectors]'
end

function sqrtm{T<:Real}(A::Symmetric{T})
    F = eigfact(A)
    vsqrt = sqrt(complex(F[:values]))
    if all(imag(vsqrt) .== 0)
        retmat = copytri!(scale(F[:vectors], real(vsqrt)) * F[:vectors]', 'U')
    else
        zc = complex(F[:vectors])
        retmat = copytri!(scale(zc, vsqrt) * zc', 'U')
    end
    retmat
end
