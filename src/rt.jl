struct RtModel{C,P,V,T, T0}
  dlogk::C
  X::T
  X0::T0
  priors::NamedTuple{P,V}
  MA1::Bool
end

function RtModel(d::Array{Vector{T}}, p::NamedTuple{P,V}) where {T, P , V}
  RtModel(d, [ones(length(ds),1) for ds in d],
          fill(ones(1), length(d)), p, true)
end

RtModel(d, x, x0, p) = RtModel(d,x,x0,p,false)

function (m::RtModel)(param)
  @unpack dlogk, X, X0, priors = m
  @unpack σR, σk, σR0, γ, ρ, α, α0 = param
  logp = logpdf(priors.σR, σR) +
    logpdf(priors.γ, γ) +
    logpdf(priors.σR0, σR0) +
    #logpdf(priors.μR0, μR0) +
    logpdf(priors.σk, σk) +
    logpdf(priors.α, α) +
    logpdf(priors.α0, α0) +
    logpdf(priors.ρ,ρ)

  # Note that this not the usual Kalman filter because the noise in
  # dlogk is an MA(1) instead of independent. We use
  # https://arxiv.org/pdf/1909.10582.pdf
  # and adopt their notation
  A = m.MA1 ? SMatrix{2,2}([2*σk^2  -σk^2; -σk^2  2*σk^2]) : SMatrix{2,2}([2*σk^2 0; 0 2*σk^2])
  S = length(dlogk)
  T = maximum(length.(dlogk))
  meanR = Vector{typeof(σR)}(undef, T)
  zhat = Vector{typeof(σR)}(undef, T)
  varR, varRprior, K, varz, zcoef = ma1kalman_variance(T, σR0^2, ρ, σR^2, γ, A)
  for s in 1:S
    z = dlogk[s] .+ γ .- γ* (X[s]*α)
    μR0 = dot(X0[s],α0)
    ma1kalman_mean!(meanR, zhat, z, ρ, γ, μR0, K, zcoef) #, X[s]*α)
    for t in 1:length(dlogk[s])
      μ = zhat[t]
      σ = sqrt(varz[t])
      logp += logpdf(Normal(μ, σ), z[t])
    end
  end
  return(logp)
end


function plotpostr(dates, dlogk, post, X, X0; Δ=1)
  k = [kalman(dlogk, p.σR, p.σk, p.σR0, dot(X0,p.α0), p.γ, p.ρ, X*p.α) for p in post];
  γ = [p.γ for p in post];
  Xa = hcat([X*p.α for p in post]...)
  meanR = hcat([x[1] for x in k]...)./Δ;
  varR = hcat([x[2] for x in k]...)./Δ^2;
  zhat = hcat([x[3] for x in k]...);
  c = "black"
  figr = plot(dates, mean(meanR, dims=2), ribbon=1.64*mean(sqrt.(varR),dims=2), color=c, fillalpha=0.2,
              linewidth=1.5, legend=:none, ylab="Rₜ")
  r=([quantile(meanR[t,:] - 1.64*sqrt.(varR[t,:]), 0.05) for t in 1:size(meanR,1)],
     [quantile(meanR[t,:] + 1.64*sqrt.(varR[t,:]), 0.95) for t in 1:size(meanR,1)])
  figr = plot!(figr, dates, zeros(length(r[1])), ribbon=(-r[1], r[2]), color=c,
               linewidth=0, ylim=nothing, fillalpha=0.2)
  T = length(dlogk)
  rt = [mean(dlogk[t]./γ .+ 1) for t in 1:T]
  lo = [quantile(dlogk[t]./γ .+ 1, 0.05) for t in 1:T]
  hi = [quantile(dlogk[t]./γ .+ 1, 0.95) for t in 1:T]
  figr = scatter!(figr, dates, rt, yerror=(rt-lo, hi-rt))
  figr = hline!(figr, [1.], color="red", linewidth=1.5, linestyle=:dash)
  return(figr)
end
