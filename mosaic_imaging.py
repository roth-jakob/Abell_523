from astropy.coordinates import SkyCoord

from functools import reduce
import resolve as rve
import nifty8 as ift
import configparser
import numpy as np

from os.path import join

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy import units as u

from sys import exit


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
    nthreads = 4
except ImportError:
    comm = None
    master = True
    nthreads = 1

cfg = configparser.ConfigParser()
# cfg.read("./abell_523_D.cfg")
cfg.read("./abell_523_CD_pol.cfg")
# cfg.read("./abell_523_CD_muli_frequnency.cfg")
# cfg.read("./abell_523_11_15_mfreq.cfg")
# cfg.read("./abell_523_CD_pol_multifreq.cfg")

path = './data/resolve/'
base = 'A523_CD_06_08_R'
# base = 'A523_CD_11_15_R'


center_ra = cfg['sky']['image center ra']
center_dec = cfg['sky']['image center dec']
center_frame = cfg['sky']['image center frame']
npix = int(cfg['sky']['space npix x'])
fov = float(cfg['sky']['space fov x'].split('d')[0]) * u.deg
output_directory = f"output/{base}_{cfg['Output']['output name']}_{npix}"
print('Output:', output_directory)


sky, sky_diffuse_operators = rve.sky_model_diffuse(cfg['sky'])
pdom, tdom, fdom, sdom = sky.target
sky_dtype = sky(ift.from_random(sky.domain)).dtype

psm = cfg['sky'].get('point sources', False)
psm = eval(psm) if isinstance(psm, str) else psm
if psm:
    print("Including: point source model")
    sky_points, additional_points = rve.sky_model_points(cfg["sky"])
    sky = sky + sky_points
    output_directory += '_ps'

# data_filenames = [join(path, f'{base}.ms_fld{ii:02}_spw00.npz')
#                   for ii in range(5, 11)]
# data_filenames = [data_filenames[0]]
# data_filenames = [join(path, f'{base}.ms_fld08_spw{jj:02}.npz')
#                   for jj in range(3)]

data_filenames = [join(path, f'{base}.ms_fld{ii:02}_spw{jj:02}.npz')
                  for ii in range(5, 11) for jj in range(3)]

# data_filenames = [join(path, f'{base}.ms_fld08_spw{jj:02}.npz')
#                   for jj in range(0, 5)]


all_obs = []
for file in data_filenames:
    obs = rve.Observation.load(file)
    obs = obs.to_double_precision()
    if pdom.shape != (4,):
        print('Restricting to stokes I')
        obs = obs.restrict_to_stokesi()
        obs = obs.average_stokesi()  # FIXME: Needs to be adjusted for polarization

    all_obs.append(obs)


sky_center = SkyCoord(center_ra, center_dec, unit=(
    u.hourangle, u.deg), frame=center_frame)
sky_beamer = rve.build_sky_beamer(
    sky.target,
    sky_center,
    all_obs,
    lambda freq, x: rve.alma_beam_func(D=25.0, d=1.0, freq=freq, x=x)
)
likelihoods = rve.build_mosaic_likelihoods(
    sky.target, sky_dtype, sky_beamer, all_obs, nthreads=nthreads)

lh = reduce(lambda x, y: x+y, likelihoods)
lh = lh @ sky_beamer @ sky


def callback(samples, i):
    print('Plotting iteration', i, 'in: ', output_directory)

    sky_mean = samples.average(sky)
    pols, ts, freqs, *_ = sky_mean.shape
    fig, axes = plt.subplots(pols, freqs, figsize=(freqs*4, pols*3))

    rve.ubik_tools.field2fits(sky_mean, join(
        output_directory, f'sky_reso_iter{i}.fits'))

    if freqs == 1:
        for poli, ax in enumerate(axes):
            f = sky_mean.val[poli, 0, 0].T
            if poli > 0:
                f = np.abs(f)

            im = ax.imshow(f, origin="lower", norm=LogNorm())
            plt.colorbar(im, ax=ax)

    elif pols == 1:
        for freqi, ax in enumerate(axes):
            im = ax.imshow(
                sky_mean.val[0, 0, freqi].T, origin="lower", norm=LogNorm())
            plt.colorbar(im, ax=ax)

    else:
        for poli, pol_axes in enumerate(axes):
            for freqi, ax in enumerate(pol_axes):
                if poli > 0:
                    f = np.abs(f)
                f = sky_mean.val[poli, 0, freqi].T
                im = ax.imshow(f, origin="lower", norm=LogNorm())
                plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if master:
        plt.savefig(f"{output_directory}/resovle_iteration_{i}.png")
    plt.close()


ic_sampling_early = ift.AbsDeltaEnergyController(
    name="Sampling (linear)", deltaE=0.05, iteration_limit=100
)
ic_sampling_late = ift.AbsDeltaEnergyController(
    name="Sampling (linear)", deltaE=0.05, iteration_limit=500
)
ic_newton_early = ift.AbsDeltaEnergyController(
    name="Newton", deltaE=0.5, convergence_level=2, iteration_limit=10
)
ic_newton_late = ift.AbsDeltaEnergyController(
    name="Newton", deltaE=0.5, convergence_level=2, iteration_limit=30
)
minimizer_early = ift.NewtonCG(ic_newton_early)
minimizer_late = ift.NewtonCG(ic_newton_late)


n_iterations = 20
def ic_sampling(i): return ic_sampling_early if i < 15 else ic_sampling_late
def minimizer(i): return minimizer_early if i < 15 else minimizer_late
def n_samples(i): return 2 if i < 10 else 4


print(output_directory)
export_operator_outputs = {
    key: val for key, val in sky_diffuse_operators.items() if 'power' not in key
}

samples = ift.optimize_kl(
    lh,
    n_iterations,
    n_samples,
    minimizer,
    ic_sampling,
    None,
    output_directory=output_directory,
    comm=comm,
    inspect_callback=callback,
    export_operator_outputs=export_operator_outputs,
    resume=True
)

sky_mean = samples.average(sky)
rve.ubik_tools.field2fits(sky_mean, join(
    output_directory, f'sky_reso_{npix}.fits'))
