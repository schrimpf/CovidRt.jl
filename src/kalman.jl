function ma1kalman_variance(T, V0, F, W, H, A)
  # Note that this not the usual Kalman filter because the noise in
  # dlogk is an MA(1) instead of independent. We use
  # https://arxiv.org/pdf/1909.10582.pdf
  # and adopt their notation
  varR = Vector{typeof(V0)}(undef,T)
  varRprior = Vector{typeof(V0)}(undef,T)
  K = Vector{typeof(V0)}(undef,T)
  varz = Vector{typeof(V0)}(undef, T)
  zcoef = Matrix{typeof(V0)}(undef,2, T)
  varZ = MMatrix{2,2,typeof(V0)}(zeros(4))
  covzZ = MVector{2, typeof(V0)}(zeros(2))
  for t in 1:T
    varZ .= 0
    if t == 1
      varRprior[t] = V0
    else
      varRprior[t] = F*varR[t-1]*F' + W
      varZ[2,2] = H*varRprior[t-1]*H'
    end
    varZ[1,1] = H*varRprior[t]*H'
    varZ .+= A
    varz[t] = varZ[1,1]
    covzZ[1] = varZ[1,2]
    ivarZ = inv(varZ)
    zcoef[:,t] .= vec(covzZ'*ivarZ)
    iL = 1/(varZ[1,1] + dot(zcoef[:,t], covzZ))
    J = varRprior[t]*H
    K[t] = J*iL
    varR[t] =  varRprior[t]-K[t]*J' # P_t
  end
  return(varR, varRprior, K, varz, zcoef)
end

function ma1kalman_mean!(x, zhat, z, F, H, x0, K, zcoef)
  T = length(z)
  elag = @MVector zeros(eltype(zcoef[1,1]*z[1]), 2)
  for t in 1:T
    xminus = F*(t==1 ? x0 : x[t-1])
    ez = H*xminus
    zhat[t] = ez
    zhat[t] += zcoef[:,t]'*elag
    x[t] = xminus + K[t]*(z[t] - zhat[t])
    elag[2] = elag[1]
    elag[1] = z[t] - H*xminus
  end
end

function ma1kalman_mean!(x, zhat, z, F, H, x0, K, zcoef, xshift)
  T = length(z)
  elag = @MVector zeros(eltype(zcoef[1,1]*z[1]), 2)
  for t in 1:T
    xminus = F*(t==1 ? x0 : x[t-1] ) + xshift[t]
    ez = H*xminus
    zhat[t] = ez
    #zhat[t] += zcoef[:,t]'*elag
    x[t] = xminus + K[t]*(z[t] - zhat[t])
    elag[2] = elag[1]
    elag[1] = z[t] - H*xminus
  end
end

function kalman(dlogk, σR, σk, σR0, μR0, γ, ρ=1, Xα=zeros(length(dlogk)); MA1=false)
  T = length(dlogk)
  A = MA1 ?  SMatrix{2,2}([2*σk^2  -σk^2; -σk^2  2*σk^2]) :  SMatrix{2,2}([2*σk^2  0; 0  2*σk^2])
  meanR = Vector{typeof(σR)}(undef, T)
  zhat = Vector{typeof(σR)}(undef, T)
  varR, varRprior, K, varz, zcoef = ma1kalman_variance(T, σR0^2, ρ, σR^2, γ, A)
  z = dlogk .+ γ .- γ* (Xα)
  ma1kalman_mean!(meanR, zhat, z, ρ, γ, μR0, K, zcoef)
  meanR .+= Xα
  zhat .+= -γ .+γ*(Xα)
  return(meanR, varR, zhat)
end
