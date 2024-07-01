using Revise
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using Parameters

@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}   
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    rng::MersenneTwister
    beacons::Array{Float64, 2}
    d::Float64
    rmin::Float64
end


function PropagateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    # println("cov0")
    # println( 𝒫.Σv)
    # predict
    μp = F * μb + a# add your code here
    Σp = F * Σb * F' + Σw # add your code here
    return MvNormal(μp, Σp)
end 



function PropagateUpdateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    # predict
    # mean/var on motion, measurement model
    μp = F * μb + a# add your code here
    Σp = F * Σb * F' + Σw # add your code here
    # update
    H = I # assuming observation model h(x) = x ->  the identity matrix
    K = Σp * H' * inv(H * Σp * H' + Σv) # Kalman gain
    μb′ = μp + K * (o - H * μp)
    Σb′ = (I - K * H) * Σp
    return MvNormal(μb′, Σb′)
end    

function SampleMotionModel(𝒫::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    #given linear motion model - X_k+1 = F * X_k + a_k + w
    F = 𝒫.F
    Σw = 𝒫.Σw
    w = rand(𝒫.rng,MvNormal([0.0, 0.0], 𝒫.Σw))
    motion_samp = F * x + a + w 
    return motion_samp
end 

function GenerateObservation(𝒫::POMDPscenario, x::Array{Float64, 1})
    Σv = 𝒫.Σv
    v = rand(𝒫.rng,MvNormal([0.0, 0.0], Σv))
    meas_samp = x + v
    return meas_samp
end   


function GenerateObservationFromBeacons(𝒫::POMDPscenario, x::Array{Float64, 1})::Union{NamedTuple, Nothing}
    distances = vec(norm.(eachcol(𝒫.beacons') .- eachcol(x)))
    for (index, distance) in enumerate(distances)
        if distance <= 𝒫.d
            # obs = x + rand(𝒫.rng,MvNormal([0.0, 0.0], I(2)*0.01)) #mean zero var sig v
            obs = x + rand(𝒫.rng,MvNormal([0.0, 0.0],(0.01 * max(distance,𝒫.rmin))^2 *I(2))) #mean zero var sig v
            # println("cov")
            # 𝒫.Σv
            return (obs=obs, index=index) 
        end    
    end 
    return nothing    
end    


function main()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    b0 = MvNormal(μ0, Σ0)
    d =1.0 
    rmin = 0.1
    # set beacons locations 
    beacons = [0.0 0.0 ; 0.0 1.0 ; 1.0 0.0 ;
               1.0 1.0; -1.0 -1.0 ; 0.5 0.5 ;
               -0.5 0.5; 0.5 -0.5; -0.5 -0.5]*9
    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                      Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                      Σv=0.01^2*[1.0 0.0; 0.0 1.0], 
                      rng = rng , beacons=beacons, d=d, rmin=rmin) 
    ak = [1.0, 0.0]
    xgt0 = [0.5, 0.2]
    T = 10
    # generating the trajectory
    τ = [xgt0]
    # generate motion trajectory
    for i in 1:T-1
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    end  
    # generate observation trajectory
    τobs = Array{Float64, 1}[]
    for i in 1:T
        push!(τobs, GenerateObservation(𝒫, τ[i]))
    end  
    
    # generate beliefs dead reckoning 
    τbp = [b0]
    
    for i in 1:T-1
        push!(τbp, PropagateBelief(τbp[end],  𝒫, ak))
    end
    
    #generate posteriors 
    τb = [b0]
    for i in 1:T-1
        push!(τb, PropagateUpdateBelief(τb[end],  𝒫, ak, τobs[i+1]))
    end


    
    # plots 
    dr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)

    for i in 1:T
        covellipse!(τbp[i].μ, τbp[i].Σ, showaxes=true, n_std=3, label="step $i")
    end
    savefig(dr,"dr.pdf")

    tr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    for i in 1:T
        covellipse!(τb[i].μ, τb[i].Σ, showaxes=true, n_std=3, label="step $i")
    end
    # scatter(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)

    savefig(tr,"tr.pdf")

    
               
    xgt0 = [-0.5, -0.2]           
    ak = [0.1, 0.1]           
    T = 100
    # generating the trajectory
    τ = [xgt0]
    # generate motion trajectory
    for i in 1:T-1
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    end 

    # generate observation trajectory
    τobsbeacons = []
    for i in 1:T
        push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[i]))
    end  

    println(τobsbeacons)
    bplot =  scatter(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    scatter!([x[1] for x in τobsbeacons if x!=nothing], [x[2] for x in τobsbeacons if x!=nothing], label="obs")
    savefig(bplot,"beacons_wO_r.pdf")

     # Update beliefs using beacon observations
     τbb = [b0]
     for i in 1:T-1
         if τobsbeacons[i+1] != nothing
             obs = τobsbeacons[i+1].obs
             push!(τbb, PropagateUpdateBelief(τbb[end], 𝒫, ak, obs))
         else
             push!(τbb, PropagateBelief(τbb[end], 𝒫, ak))
         end
     end
    
     # Plot results
     trb = scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
     for i in 1:T
         covellipse!( τbb[i].μ, τbb[i].Σ, showaxes=true, n_std=3, label="step $i")
     end
     scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
     savefig(trb, "trb.pdf")
    
    gt = [xgt0]
    loc_err = []
    for i in 1:T-1
        push!(gt, ak*i)
        push!(loc_err, τ[i] .- gt[i])
        # println(τ[i])
        # println(loc_err)
    end
    println(loc_err)
    # loc_err_plt = scatter(loc_err[1,:] , loc_err[2,:])
    loc_err_plt = scatter([x[1] for x in loc_err], [x[2] for x in loc_err], label="err")

    savefig(loc_err_plt,"loc_err_plt_var_noise.pdf")

    
    
    
    
    # use function det(b.Σ) to calculate determinant of the matrix
    # det(b.Σ)
    # print("2(a) : ")
end 

main()