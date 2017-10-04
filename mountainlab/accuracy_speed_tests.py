import numpy as np
import finufftpy
import math
import time

def compute_error(Xest,Xtrue):
    numer=np.sqrt(np.sum(np.abs((Xest-Xtrue))**2));
    denom=np.sqrt(np.sum(np.abs((Xest+Xtrue)/2))**2);
    if (denom!=0):
        return numer/denom
    else:
        return 0

def print_report(label,elapsed,Xest,Xtrue,npts):
    print(label+':')
    print('    Est. error      %g' % (compute_error(Xest,Xtrue)))
    print('    Elapsed (sec)   %g' % (elapsed))
    print('    nu.pts/sec      %g' % (npts/elapsed))
    print('')

def accuracy_speed_tests(num_nonuniform_points,num_uniform_points,eps,num_trials):
    print("test")
    nj,nk = int(num_nonuniform_points),int(num_nonuniform_points)
    iflag=1
    num_samples=int(np.minimum(20,num_uniform_points*0.5+1)) #for estimating accuracy

    print('Accuracy and speed tests for %d nonuniform points and eps=%g (error estimates use %d samples per run)' % (num_nonuniform_points,eps,num_samples))

    # for doing the error estimates
    Xest=np.zeros(num_samples,dtype=np.complex128)
    Xtrue=np.zeros(num_samples,dtype=np.complex128)

    ###### 1-d
    ms=int(num_uniform_points)

    print('Generating sample data...')
    timer=time.time()
    xj=np.random.rand(nj)*2*math.pi-math.pi
    cj=np.random.rand(nj)+1j*np.random.rand(nj);
    sk=np.random.rand(nj)*2*math.pi-math.pi
    fk=np.random.rand(ms)+1j*np.random.rand(ms);
    print('Elapsed time for generating sample data: %g sec' % (time.time()-timer))
    
    fk_out=np.zeros([ms],dtype=np.complex128)
    timer=time.time()
    for trial in np.arange(0,num_trials):
        ret=finufftpy.finufft1d1(xj,cj,iflag,eps,ms,fk_out)
    elapsed=time.time()-timer

    k=np.arange(-np.floor(ms/2),np.floor((ms-1)/2+1))
    for ii in np.arange(0,num_samples):
        Xtrue[ii]=np.sum(cj * np.exp(1j*k[ii]*xj))
        Xest[ii]=fk_out[ii]
    print_report('finufft1d1',elapsed,Xest,Xtrue,nj*num_trials)

    cj_out=np.zeros([nj],dtype=np.complex128);
    timer=time.time()
    for trial in np.arange(0,num_trials):
        ret=finufftpy.finufft1d2(xj,cj_out,iflag,eps,ms,fk)
    elapsed=time.time()-timer

    k=np.arange(-np.floor(ms/2),np.floor((ms-1)/2+1))
    for ii in np.arange(0,num_samples):
        Xtrue[ii]=np.sum(fk * np.exp(1j*k*xj[ii]))
        Xest[ii]=cj_out[ii]
    print_report('finufft1d2',elapsed,Xest,Xtrue,nj*num_trials)

    fk_out=np.zeros([nk],dtype=np.complex128)
    timer=time.time()
    for trial in np.arange(0,num_trials):
        ret=finufftpy.finufft1d3(xj,cj,iflag,eps,sk,fk_out)
    elapsed=time.time()-timer

    for ii in np.arange(0,num_samples):
        Xtrue[ii]=np.sum(cj * np.exp(1j*sk[ii]*xj))
        Xest[ii]=fk_out[ii]
    print_report('finufft1d3',elapsed,Xest,Xtrue,(nj+nk)*num_trials)

    ###### 2-d

    ms=int(np.ceil(num_uniform_points**(1/2)))
    mt=ms

    print('Generating sample data...')
    timer=time.time()
    yj=np.random.rand(nj)*2*math.pi-math.pi
    tk=np.random.rand(nj)*2*math.pi-math.pi
    fk=np.random.rand(ms,mt)+1j*np.random.rand(ms,mt);
    print('Elapsed time for generating sample data: %g sec' % (time.time()-timer))    

    fk_out=np.zeros([ms,mt],dtype=np.complex128,order='F')
    timer=time.time()
    for trial in np.arange(0,num_trials):
        ret=finufftpy.finufft2d1(xj,yj,cj,iflag,eps,ms,mt,fk_out)
    elapsed=time.time()-timer

    Ks,Kt=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1)]
    fk_out_vec=fk_out.ravel()
    Ks_vec=Ks.ravel()
    Kt_vec=Kt.ravel()
    for ii in np.arange(0,num_samples):
        Xtrue[ii]=np.sum(cj * np.exp(1j*(Ks_vec[ii]*xj+Kt_vec[ii]*yj)))
        Xest[ii]=fk_out_vec[ii]
    print_report('finufft2d1',elapsed,Xest,Xtrue,nj*num_trials)

    cj_out=np.zeros([nj],dtype=np.complex128);
    timer=time.time()
    for trial in np.arange(0,num_trials):
        ret=finufftpy.finufft2d2(xj,yj,cj_out,iflag,eps,ms,mt,fk)
    elapsed=time.time()-timer

    Ks,Kt=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1)]
    for ii in np.arange(0,num_samples):
        Xtrue[ii]=np.sum(fk * np.exp(1j*(Ks*xj[ii]+Kt*yj[ii])))
        Xest[ii]=cj_out[ii]
    print_report('finufft2d2',elapsed,Xest,Xtrue,nj*num_trials)

    fk_out=np.zeros([nk],dtype=np.complex128)
    timer=time.time()
    for trial in np.arange(0,num_trials):
        ret=finufftpy.finufft2d3(xj,yj,cj,iflag,eps,sk,tk,fk_out)
    elapsed=time.time()-timer

    for ii in np.arange(0,num_samples):
        Xtrue[ii]=np.sum(cj * np.exp(1j*(sk[ii]*xj+tk[ii]*yj)))
        Xest[ii]=fk_out[ii]
    print_report('finufft2d3',elapsed,Xest,Xtrue,(nj+nk)*num_trials)

    ###### 3-d
    ms=int(np.ceil(num_uniform_points**(1/3)))
    mt=ms
    mu=ms
    
    print('Generating sample data...')
    timer=time.time()
    zj=np.random.rand(nj)*2*math.pi-math.pi
    uk=np.random.rand(nj)*2*math.pi-math.pi
    fk=np.random.rand(ms,mt,mu)+1j*np.random.rand(ms,mt,mu);
    print('Elapsed time for generating sample data: %g sec' % (time.time()-timer))    

    fk_out=np.zeros([ms,mt,mu],dtype=np.complex128,order='F')
    timer=time.time()
    for trial in np.arange(0,num_trials):
        ret=finufftpy.finufft3d1(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk_out)
    elapsed=time.time()-timer

    Ks,Kt,Ku=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1),-np.floor(mu/2):np.floor((mu-1)/2+1)]
    fk_out_vec=fk_out.ravel()
    Ks_vec=Ks.ravel()
    Kt_vec=Kt.ravel()
    Ku_vec=Ku.ravel()
    for ii in np.arange(0,num_samples):
        Xtrue[ii]=np.sum(cj * np.exp(1j*(Ks_vec[ii]*xj+Kt_vec[ii]*yj+Ku_vec[ii]*zj)))
        Xest[ii]=fk_out_vec[ii]
    print_report('finufft3d1',elapsed,Xest,Xtrue,nj*num_trials)

    cj_out=np.zeros([nj],dtype=np.complex128);
    timer=time.time()
    for trial in np.arange(0,num_trials):
        ret=finufftpy.finufft3d2(xj,yj,zj,cj_out,iflag,eps,ms,mt,mu,fk)
    elapsed=time.time()-timer

    Ks,Kt,Ku=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1),-np.floor(mu/2):np.floor((mu-1)/2+1)]
    for ii in np.arange(0,num_samples):
        Xtrue[ii]=np.sum(fk * np.exp(1j*(Ks*xj[ii]+Kt*yj[ii]+Ku*zj[ii])))
        Xest[ii]=cj_out[ii]
    print_report('finufft3d2',elapsed,Xest,Xtrue,nj*num_trials)

    fk_out=np.zeros([nk],dtype=np.complex128)
    timer=time.time()
    for trial in np.arange(0,num_trials):
        ret=finufftpy.finufft3d3(xj,yj,zj,cj,iflag,eps,sk,tk,uk,fk_out)
    elapsed=time.time()-timer

    for ii in np.arange(0,num_samples):
        Xtrue[ii]=np.sum(cj * np.exp(1j*(sk[ii]*xj+tk[ii]*yj+uk[ii]*zj)))
        Xest[ii]=fk_out[ii]
    print_report('finufft3d3',elapsed,Xest,Xtrue,(nj+nk)*num_trials)