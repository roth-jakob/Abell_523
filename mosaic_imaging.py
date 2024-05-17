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
except ImportError:
    comm = None
    master = True

cfg = configparser.ConfigParser()
cfg.read("./abell_523_D.cfg")
# cfg.read("./abell_523_CD_pol.cfg")
# cfg.read("./abell_523_CD_muli_frequnency.cfg")
# cfg.read("abell_523_CD_muli_frequnency.cfg")
path = '/home/jruestig/pro/python/Abell_523/data/resolve/'
base = 'A523_CD_06_08_R'


center_ra = cfg['sky']['image center ra']
center_dec = cfg['sky']['image center dec']
center_frame = cfg['sky']['image center frame']
npix = int(cfg['sky']['space npix x'])
fov = float(cfg['sky']['space fov x'].split('d')[0]) * u.deg
output_directory = f"output/{base}_{cfg['Output']['output name']}_{npix}"

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

data_filenames = [join(path, f'{base}.ms_fld{ii:02}_spw00.npz')
                  for ii in range(5, 11)]
# data_filenames = [data_filenames[0]]
# data_filenames = [join(path, f'{base}.ms_fld08_spw{jj:02}.npz')
#                   for jj in range(3)]


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


def build_response(field_key, dx, dy, obs, domain, sky_dtype):
    R = rve.InterferometryResponse(
        obs, domain, False, 1e-3, center_x=dx, center_y=dy)

    FIELD_EXTRACTOR = ift.JaxLinearOperator(
        sky_beamer.target,
        R.domain,
        lambda x: x[field_key],
        domain_dtype={k: sky_dtype for k, v in sky_beamer.target.items()}
    )

    return R @ FIELD_EXTRACTOR


def build_energy(response, obs):
    return rve.DiagonalGaussianLikelihood(
        data=obs.vis,
        inverse_covariance=obs.weight,
        mask=obs.mask
    ) @ response


responses = []
energies = []
for kk, vv, obs in zip(
    sky_beamer._beam_directions.keys(),
    sky_beamer._beam_directions.values(),
    all_obs
):
    # R = build_response(kk, vv['dx'], vv['dy'], obs, tmp_sky.dtype)
    R = build_response(
        field_key=kk, dx=vv['dx'], dy=vv['dy'], obs=obs, domain=sky.target,
        sky_dtype=sky_dtype)
    responses.append(R @ sky_beamer)
    energies.append(build_energy(R, obs))


for ii in range(1):
    rnd = ift.from_random(sky.domain)
    f = sky(rnd)

    tot_response_adjoint = np.zeros_like(f.val)
    for ii, (rr, oo) in enumerate(zip(responses, all_obs)):
        dirty = rr.adjoint(oo.vis).val
        plt.imshow(dirty[0, 0, 0], origin='lower')
        plt.title(f'fldid={ii+5}')
        plt.show()
        tot_response_adjoint += dirty
    plt.imshow(tot_response_adjoint[0, 0, 0], origin='lower')
    plt.show()

lh = reduce(lambda x, y: x+y, energies)
lh = lh @ sky_beamer @ sky


def callback(samples, i):
    sky_mean = samples.average(sky)
    plt.imshow(sky_mean.val[0, 0, 0].T, origin="lower", norm=LogNorm())
    # plt.contour(beam, levels=[0.1], colors='white')
    plt.colorbar()
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


n_iterations = 7
def ic_sampling(i): return ic_sampling_early if i < 15 else ic_sampling_late
def minimizer(i): return minimizer_early if i < 15 else minimizer_late
def n_samples(i): return 2 if i < 7 else 4


print(output_directory)
export_operator_outputs = {
    key: val for key, val in sky_diffuse_operators.items() if 'power' not in key}

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
