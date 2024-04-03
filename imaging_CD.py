from numpy.core.numerictypes import obj2sctype
import nifty8 as ift
import resolve as rve

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import configparser

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
except ImportError:
    comm = None
    master = True

ift.random.push_sseq_from_seed(42)

obs_all = list(rve.ms2observations_all("A523_CD_06-08_MC_R.ms", "DATA"))

# obs = obs_all[1]
obs_all = [obs.to_double_precision() for obs in obs_all]
obs_all = [obs.restrict_to_stokesi() for obs in obs_all]

cfg = configparser.ConfigParser()
# cfg.read("abell_523_C.cfg")
cfg.read("abell_523_CD_muli_frequnency.cfg")

sky_diffuse, additional_diffuse = rve.sky_model_diffuse(cfg["sky"], obs_all)
# sky_points, additional_points = rve.sky_model_points(cfg["sky"])
# sky = sky_diffuse + sky_points
sky = sky_diffuse
# import numpy as np
# for i in range(10):
#     rnd = ift.from_random(sky.domain)
#     smp = sky_points.force(rnd).val
#     loc = np.where(smp[0,0,0,:,:]>0.001, 1, 0)
#     # ift.single_plot(smp, norm=LogNorm())
#     plt.imshow(loc.T, origin='lower')
#     plt.colorbar()
#     plt.show()

lh = rve.ImagingLikelihood(obs_all, sky, 1e-7, False, nthreads=4)

def callback(samples, i):
    sky_mean = samples.average(sky)
    for n in range(sky_mean.shape[2]):
        plt.imshow(sky_mean.val[0, 0, n, :, :].T, origin="lower", norm=LogNorm())
        plt.colorbar()
        if master:
            plt.savefig(f"resovle_iteration_{i}_band_{n}.png")
        plt.close()
    rve.ubik_tools.field2fits(sky_mean, observations=obs_all, file_name=f'abell_523_resolve_iteration_{i}.fits')


ic_sampling_early = ift.AbsDeltaEnergyController(
    name="Sampling (linear)", deltaE=0.005, iteration_limit=100
)
ic_sampling_late = ift.AbsDeltaEnergyController(
    name="Sampling (linear)", deltaE=0.005, iteration_limit=500
)
ic_newton_early = ift.AbsDeltaEnergyController(
    name="Newton", deltaE=0.5, convergence_level=2, iteration_limit=10
)
ic_newton_late = ift.AbsDeltaEnergyController(
    name="Newton", deltaE=0.5, convergence_level=2, iteration_limit=30
)
minimizer_early = ift.NewtonCG(ic_newton_early)
minimizer_late = ift.NewtonCG(ic_newton_late)


n_iterations = 30
ic_sampling = lambda i: ic_sampling_early if i < 15 else ic_sampling_late
minimizer = lambda i: minimizer_early if i < 15 else minimizer_late
n_samples = lambda i: 2 if i < 20 else 4

samples = ift.optimize_kl(
    lh,
    n_iterations,
    n_samples,
    minimizer,
    ic_sampling,
    None,
    output_directory="resolve_demo",
    comm=comm,
    inspect_callback=callback,

)
