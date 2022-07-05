import dedalus.public as de
from .helpers import *

nz = 128 # number of Chebyshev modes for EVP

def bottom_frontal_zone_instability(
        k, l, M, Hbfz=Hbfz,
        tht=θ, f=f,
        kap0=κ0, kap1=κ1, h=h, Pr=σ,
        nz=nz, H=Hbfz*1.5, δz=δz,
        nuh=0., nu4=0., kaph=0., kap4=0.,
        nh=0.
    ):
    
    ## build domain
    zbasis = de.Chebyshev('z', nz, interval=(0, H), dealias=3/2)
    domain = de.Domain([zbasis], np.complex128)
    z = domain.grid(0)
    
    ## Fields
    # state variables
    B = domain.new_field(name='B')
    U = domain.new_field(name='U')
    V = domain.new_field(name='V')
    Bz = domain.new_field(name='Bz')
    Uz = domain.new_field(name='Uz')
    Vz = domain.new_field(name='Vz')
    
    # basic state
    B['g'], U['g'], V['g'], Bz['g'], Uz['g'], Vz['g'] = bottom_frontal_zone_basic(z, M, Hbfz=Hbfz, θ=tht, f=f, δz=δz, derivatives=True)
    
    # non-constant coefficients
    kap = domain.new_field(name='kap')
    kap['g'] = kap0 + kap1*np.exp(-z/h) # slope-normal diffusivity
    
    # setup problem
    problem = de.EVP(
        domain,
        variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz', 'bz'], # perturbation variables
        eigenvalue='omg',
        tolerance = 1e-12
    )
    
    problem.parameters['k'] = k
    problem.parameters['l'] = l
    
    problem.parameters['tht'] = tht
    problem.parameters['N'] = N
    problem.parameters['f'] = f
    problem.parameters['kap'] = kap
    problem.parameters['Pr'] = Pr
    
    problem.parameters['nuh'] = nuh
    problem.parameters['nu4'] = nu4
    problem.parameters['kaph'] = kaph
    problem.parameters['kap4'] = kap4
    
    problem.parameters['nh'] = nh
    
    problem.parameters['B'] = B
    problem.parameters['U'] = U
    problem.parameters['V'] = V
    problem.parameters['Bz'] = Bz
    problem.parameters['Uz'] = Uz
    problem.parameters['Vz'] = Vz

    # substitutions
    problem.substitutions['dx(A)'] = "1j*k*A"
    problem.substitutions['dy(A)'] = "1j*l*A"
    problem.substitutions['dt(A)'] = "-1j*omg*A"

#     # Linearized equation set
#     problem.add_equation(# Cross-slope momentum
#         'dt(u) + w*Uz + U*dx(u) + V*dy(u) - f*v*cos(tht) + dx(p)'
#         '- b*sin(tht)'
#         '- Pr *( dz(kap)*uz  + kap*dz(uz)  )'
#         '- nuh*( dx(dx(u)) + dy(dy(u)) )'
#         '+ nu4*( dx(dx(dx(dx(u)))) + 2*dx(dx(dy(dy(u)))) + dy(dy(dy(dy(u)))) )'
#         '= 0'
#     )
#     problem.add_equation((# Along-slope momentum
#         'dt(v) + w*Vz + U*dx(v) + V*dy(v) + f*(u*cos(tht) - w*sin(tht)) + dy(p)'
#         '- Pr* ( dz(kap)*vz  + kap*dz(vz)  )'
#         '- nuh*( dx(dx(v)) + dy(dy(v)) )'
#         '+ nu4*( dx(dx(dx(dx(v)))) + 2*dx(dx(dy(dy(v)))) + dy(dy(dy(dy(v)))) )'
#         '= 0'
#     ))
#     problem.add_equation((# Slope-normal momentum
#         'nh*(dt(w) + U*dx(w) + V*dy(w) + f*v*sin(tht)'
#         '- Pr* ( dz(kap)*wz  + kap*dz(wz)  ) )'
#         '+ dz(p) - b*cos(tht) '
#         '= 0'
#     ))
#     problem.add_equation((# Buoyancy
#         'dt(b) + u*N**2*sin(tht) + w*N**2*cos(tht) + w*Bz + U*dx(b) + V*dy(b)'
#         '-    ( dz(kap)*bz  + kap*dz(bz)   )'
#         '- kaph*( dx(dx(b)) + dy(dy(b)) )'
#         '+ kap4*( dx(dx(dx(dx(b)))) + 2*dx(dx(dy(dy(b)))) + dy(dy(dy(dy(b)))) )'
#         '= 0'
#     ))

    # Linearized equation set
    problem.add_equation(# Cross-slope momentum
        'dt(u) + U*dx(u) + V*dy(u) - f*v*cos(tht) + dx(p)'
        '- b*sin(tht)'
        '- Pr *( dz(kap)*uz  + kap*dz(uz)  )'
        '- nuh*( dx(dx(u)) + dy(dy(u)) )'
        '+ nu4*( dx(dx(dx(dx(u)))) + 2*dx(dx(dy(dy(u)))) + dy(dy(dy(dy(u)))) )'
        '= 0'
    )
    problem.add_equation((# Along-slope momentum
        'dt(v) + w*Vz + U*dx(v) + V*dy(v) + f*(u*cos(tht) - w*sin(tht)) + dy(p)'
        '- Pr* ( dz(kap)*vz  + kap*dz(vz)  )'
        '- nuh*( dx(dx(v)) + dy(dy(v)) )'
        '+ nu4*( dx(dx(dx(dx(v)))) + 2*dx(dx(dy(dy(v)))) + dy(dy(dy(dy(v)))) )'
        '= 0'
    ))
    problem.add_equation((# Slope-normal momentum
        'nh*(dt(w) + U*dx(w) + V*dy(w) + f*v*sin(tht)'
        '- Pr* ( dz(kap)*wz  + kap*dz(wz)  ) )'
        '+ dz(p) - b*cos(tht) '
        '= 0'
    ))
    problem.add_equation((# Buoyancy
        'dt(b) + u*N**2*sin(tht) + w*N**2*cos(tht) + w*Bz + U*dx(b) + V*dy(b)'
        '-    ( dz(kap)*bz  + kap*dz(bz)   )'
        '- kaph*( dx(dx(b)) + dy(dy(b)) )'
        '+ kap4*( dx(dx(dx(dx(b)))) + 2*dx(dx(dy(dy(b)))) + dy(dy(dy(dy(b)))) )'
        '= 0'
    ))

    problem.add_equation('dx(u) + dy(v) + wz = 0')
    
    # substitutions to turn second-order problem into first-order problem
    problem.add_equation('uz - dz(u) = 0')
    problem.add_equation('vz - dz(v) = 0')
    problem.add_equation('wz - dz(w) = 0')
    problem.add_equation('bz - dz(b) = 0')

    # bottom boundary conditions
    problem.add_equation('left(u) = 0')
    problem.add_equation('left(v) = 0')
    problem.add_equation('left(w) = 0')
    problem.add_equation('left(bz) = 0')

    problem.add_equation('right(uz) = 0')
    problem.add_equation('right(vz) = 0')
    problem.add_equation('right(w) = - right(u)*tan(tht)') # vanishing vertical (not slope-normal) velocity
    problem.add_equation('right(bz) = 0')
    
    # set up solver
    solver = problem.build_solver()

    # solve the EVP
    solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)

    # sort eigenvalues
    omega = np.copy(solver.eigenvalues)
    omega[np.isnan(omega)] = 0.
    omega[np.isinf(omega)] = 0.
    idx = np.argsort(omega.imag)[-1] # sorts from small to large
    
    solver.set_state(idx)

    # collect eigenvector
    u = solver.state['u']
    v = solver.state['v']
    w = solver.state['w']
    b = solver.state['b']
    bz = solver.state['bz']
    uz = solver.state['uz']
    vz = solver.state['vz']
    
    U.set_scales(1)
    V.set_scales(1)
    kap.set_scales(1)
    
    return {
        'b':b['g'], 'u':u['g'], 'v':v['g'], 'w':w['g'], 'bz':bz['g'], 'uz':uz['g'], 'vz':vz['g'], 'omega':omega, 'idx':idx,
        'B':B['g'], 'U':U['g'], 'V':V['g'], 'Bz':Bz['g'], 'Uz':Uz['g'], 'Vz':Vz['g'], 'kap':kap['g'], 'z':z,
        'problem':problem, 'solver':solver,
    }
