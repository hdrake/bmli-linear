import numpy as np
import scipy
import dedalus.public as d3

N = 1.e-3 # buoyancy frequency
f = 5.e-5 # Coriolis parameter
θ = 2.e-3 # slope angle

Hbfz = 300. # thickness of bottom frontal zone
M = 4.0e-5 # strength of bottom front
δz = 25. # interface smoothing scale

κ0 = 5.e-5 # background diffusivity
κ1 = 5.e-3 # bottom enhancement of diffusivity
h = 250. # decay scale of mixing
σ = 1 # Prandtl number

H = 1500. # domain height
nz = 128 # number of Chebyshev modes for EVP

def sigmoid(z, z0=Hbfz, δz=δz):
    return (scipy.special.erf(np.sqrt(np.pi)/2. * -(z-z0)/ δz ) + 1.)/2.

def sigmoid_int(z, z0=Hbfz, δz=δz):
    return ( (z-z0)*(scipy.special.erf(np.sqrt(np.pi)/2. * -(z-z0)/ δz ) + 1.) - 2/np.pi*δz*np.exp(-np.pi*(z-z0)**2/(2*δz)**2) )/2.

def bottom_frontal_zone_basic(z, M, Hbfz=Hbfz, θ=θ, f=f, δz=δz):
    B = np.zeros_like(z)
    U = np.zeros_like(z)
    V = np.zeros_like(z)
    
    B = -M**2/np.sin(θ) * sigmoid_int(z, z0=Hbfz, δz=δz)
    V =  M**2 / (f * np.cos(θ)) * sigmoid_int(z, z0=Hbfz, δz=δz)
    V -= np.min(V)
    
    return B, U, V

def bottom_frontal_zone_instability(
        k, l, M, Hbfz=Hbfz,
        θ=θ, f=f,
        κ0=κ0, κ1=κ1, h=h, σ=σ,
        nz=nz, H=H, δz=δz,
        νh=0., νh4=0., κh=0., κh4=0.
    ):
    
    ## Coordinates and basis
    zcoord = d3.Coordinate('z')
    dist = d3.Distributor(zcoord, dtype=np.complex128)

    zbasis = d3.Chebyshev(zcoord, size=nz, bounds=(0, H), dealias=3/2)
    
    ## Fields
    # state variables
    B = dist.Field(name='B', bases=zbasis)
    U = dist.Field(name='U', bases=zbasis)
    V = dist.Field(name='V', bases=zbasis)

    # non-constant coefficients
    κ = dist.Field(name='κ', bases=zbasis)
    z = dist.local_grid(zbasis)
    κ['g'] = κ0 + κ1*np.exp(-z/h) # slope-normal diffusivity
    
    # basic state
    B['g'], U['g'], V['g'] = bottom_frontal_zone_basic(z, M, Hbfz=Hbfz, θ=θ, f=f, δz=δz)
    
    # boundary conditions (tau method)
    τ_B = dist.Field(name="τ_B")
    τ_U = dist.Field(name="τ_U")
    τ_V = dist.Field(name="τ_V")

    τ_Bz = dist.Field(name="τ_Bz")
    τ_Uz = dist.Field(name="τ_Uz")
    τ_Vz = dist.Field(name="τ_Vz")
    
    # substitutions
    dz = lambda A: d3.Differentiate(A, zcoord)

    cosθ = np.cos(θ)
    sinθ = np.sin(θ)
    N2 = N**2

    lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    Uz = dz(U) + lift(τ_Uz)
    Vz = dz(V) + lift(τ_Vz)
    Bz = dz(B) + lift(τ_Bz)

    noflux = dist.Field(name='noflux')
    noflux['g'] = -N2*cosθ
    
    # LINEAR STABILITY ANALYSIS

    # variables
    b = dist.Field(name='b', bases=zbasis)
    u = dist.Field(name='u', bases=zbasis)
    v = dist.Field(name='v', bases=zbasis)
    w = dist.Field(name='w', bases=zbasis)
    p = dist.Field(name='p', bases=zbasis)

    # boundary condition fields (via tau method)
    τ_b = dist.Field(name="τ_b")
    τ_u = dist.Field(name="τ_u")
    τ_v = dist.Field(name="τ_v")
    τ_w = dist.Field(name="τ_w")

    τ_bz = dist.Field(name="τ_bz")
    τ_uz = dist.Field(name="τ_uz")
    τ_vz = dist.Field(name="τ_vz")
    τ_wz = dist.Field(name="τ_wz")

    # eigenvalues
    ω = dist.Field(name="ω")

    # substitutions
    dx = lambda ϕ: 1j*k*ϕ
    dy = lambda ϕ: 1j*l*ϕ
    dt = lambda ϕ: -1j*ω*ϕ
    integ = lambda A: d3.Integrate(A, 'z')

    ## How should I be setting these?
    lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    # First-order reductions
    uz = dz(u) + lift(τ_uz)
    vz = dz(v) + lift(τ_vz)
    wz = dz(w) + lift(τ_wz)
    bz = dz(b) + lift(τ_bz)

    # setup problem
    problem = d3.EVP(
        [u, v, w, b, p, # perturbation variables
         τ_u, τ_v, τ_w, τ_b, τ_uz, τ_vz, τ_wz, τ_bz], # taus for enforcing boundary conditions
        eigenvalue=ω,
        namespace=locals()
    )
    problem.add_equation(# Cross-slope momentum
        'dt(u) + w*dz(U) + U*dx(u) + V*dy(u) - f*v*cosθ + dx(p)'
        '- b*sinθ - σ*dz(κ*uz)'
        '- νh*( dx(dx(u)) + dy(dy(u)) ) + νh4*( dx(dx(dx(dx(u)))) + dy(dy(dy(dy(u)))) )'
        '+ lift(τ_u) = 0'
    )
    problem.add_equation((# Along-slope momentum
        'dt(v) + w*dz(V) + U*dx(v) + V*dy(v) + f*(u*cosθ - w*sinθ) + dy(p)'
        '- σ*dz(κ*vz)'
        '- νh*( dx(dx(v)) + dy(dy(v)) ) + νh4*( dx(dx(dx(dx(v)))) + dy(dy(dy(dy(v)))) )'
        '+ lift(τ_v) = 0'
    ))
    problem.add_equation((# Slope-normal momentum
        'dt(w) + U*dx(w) + V*dy(w) + f*v*sinθ + dz(p)'
        '- b*cosθ - σ*dz(κ*wz)'
        '+ lift(τ_w) = 0'
    ))
    problem.add_equation((# Buoyancy
        'dt(b) + u*N2*sinθ + w*N2*cosθ + w*dz(B) + U*dx(b) + V*dy(b)'
        '- dz(κ*bz)'
        '- κh*( dx(dx(b)) + dy(dy(b)) ) + κh4*( dx(dx(dx(dx(b)))) + dy(dy(dy(dy(b)))) )'
        '+ lift(τ_b) = 0'
    ))
    problem.add_equation('dx(u) + dy(v) + wz = 0')

    problem.add_equation('u(z=0) = 0')
    problem.add_equation('v(z=0) = 0')
    problem.add_equation('w(z=0) = 0')
    problem.add_equation('dz(b)(z=0) = 0')

    problem.add_equation('dz(u)(z=H) = 0')
    problem.add_equation('dz(v)(z=H) = 0')
    problem.add_equation('w(z=H)*cosθ + u(z=H)*sinθ = 0') # vanishing vertical (not slope-normal) velocity
    problem.add_equation('dz(b)(z=H) = 0')

    # set up solver
    solver = problem.build_solver(ncc_cutoff=1e-10, entry_cutoff=0)

    # solve the EVP
    solver.solve_dense(solver.subproblems[0])

    # sort eigenvalues
    omega = np.copy(solver.eigenvalues)
    omega[np.isnan(omega)] = 0.
    omega[np.isinf(omega)] = 0.
    idx = np.argsort(omega.imag)[-1] # sorts from small to large
    
    solver.set_state(idx, solver.subsystems[0])
    
    B.change_scales(1)
    U.change_scales(1)
    V.change_scales(1)

    return {
        'b':b, 'u':u, 'v':v, 'w':w, 'omega':omega, 'idx':idx,
        'B':B, 'U':U, 'V':V, 'Bz':Bz, 'Uz':Uz, 'Vz':Vz, 'κ':κ, 'z':z,
        'problem':problem, 'solver':solver,
    }

