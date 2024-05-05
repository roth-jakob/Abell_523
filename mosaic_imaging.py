from typing import List, Union, Tuple

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.units import Unit
from astropy import units


from functools import reduce
from resolve.library.primary_beams import alma_beam_func
import resolve as rve
import nifty8 as ift
import configparser
import numpy as np

from beamer import SkyBeamer

from os.path import join

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy import units as u


def build_astropy_wcs(
    center: SkyCoord,
    shape: Tuple[int, int],
    fov: Tuple[Unit, Unit],
    rotation: Unit = 0.0 * units.deg,
) -> WCS:
    '''
    Specify the Astropy wcs.

    Parameters
    ----------
    center : SkyCoord
        The value of the center of the coordinate system (crval).

    shape : tuple
        The shape of the grid.

    fov : tuple
        The field of view of the grid. Typically given in degrees.

    rotation : units.Quantity
        The rotation of the grid WCS with respect to the ICRS system, in degrees.
    '''

    # Create a WCS object
    w = WCS(naxis=2)

    # Rotation
    rotation_value = rotation.to(units.rad).value
    pc11 = np.cos(rotation_value)
    pc12 = -np.sin(rotation_value)
    pc21 = np.sin(rotation_value)
    pc22 = np.cos(rotation_value)

    # Set up ICRS system
    w.wcs.crpix = [shape[0] / 2 + 0.5, shape[1] / 2 + 0.5]
    w.wcs.cdelt = [-fov[0].to(units.deg).value / shape[0],
                   fov[1].to(units.deg).value / shape[1]]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.pc = np.array([[pc11, pc12], [pc21, pc22]])

    return w


try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
except ImportError:
    comm = None
    master = True

path = '/home/jruestig/pro/python/resolve/demo/multi_resolve/JVLA_data/resolve'
data_filenames = [
    join(path, f'A523_D_06-08.ms_fld{ii:02}_spw1.npz') for ii in range(5, 11)]
data_filenames = [data_filenames[0]]


all_obs = []
for file in data_filenames:
    obs = rve.Observation.load(file)
    obs = obs.to_double_precision()
    obs = obs.restrict_to_stokesi()
    obs = obs.average_stokesi()  # FIXME: Needs to be adjusted for polarization
    all_obs.append(obs)


cfg = configparser.ConfigParser()
cfg.read("./abell_523_D.cfg")
sky, sky_diffuse_operators = rve.sky_model_diffuse(cfg['sky'])
sdom = sky.target[3]

center_ra = cfg['sky']['image center ra']
center_dec = cfg['sky']['image center dec']
center_frame = cfg['sky']['image center frame']
npix = int(cfg['sky']['space npix x'])
fov = float(cfg['sky']['space fov x'].split('d')[0]) * u.deg

sky_center = SkyCoord(center_ra, center_dec, unit=(
    u.hourangle, u.deg), frame=center_frame)
wcs = build_astropy_wcs(sky_center, (npix,)*2, (fov,)*2)
index = np.meshgrid(np.arange(npix), np.arange(npix))
sky_coords = wcs.pixel_to_world(*index)


# x_direction = coords(sdom.shape[0], sdom.distances[0])
# y_direction = coords(sdom.shape[1], sdom.distances[1])
# sky_coordinates = np.array(np.meshgrid(
#     -x_direction, y_direction, indexing='xy'))

output_directory = f"output/abell_523_D_{npix}_fld5_wcs"


beam_directions = {}
for fldid, oo in enumerate(all_obs):

    # Calculate phase center
    o_phase_center = SkyCoord(oo.direction.phase_center[0]*u.rad,
                              oo.direction.phase_center[1]*u.rad,
                              frame=center_frame)

    r = sky_center.separation(o_phase_center)
    phi = sky_center.position_angle(o_phase_center)
    dy = r.to(u.rad).value * np.cos(phi.to(u.rad).value)
    dx = r.to(u.rad).value * np.sin(phi.to(u.rad).value)

    x = sky_coords.separation(o_phase_center)
    x = x.to(u.rad).value

    print(f'Field {fldid}',
          f'Resol {oo.direction.phase_center}',
          f'Phase {o_phase_center.ra.hour}, { o_phase_center.dec.deg}',
          dx, dy)

    # x = np.sqrt((sky_coordinates[0] - dx)**2 + (sky_coordinates[1] - dy)**2)
    # beam0 = rve.vla_beam_func(freq=oo.freq.mean(), x=x)
    # beam = np.ones_like(x)
    beam = rve.alma_beam_func(D=25.0, d=1.0, freq=oo.freq.mean(), x=x)
    plt.imshow(beam, origin='lower')
    plt.show()

    beam = ift.makeField(sdom, beam)
    beam_direction = f'fld{fldid}'
    beam_directions[beam_direction] = dict(
        dx=dx,
        dy=dy,
        beam=beam
    )


# Used for the dtypes
tmp_sky = sky(ift.from_random(sky.domain))
SKY_BEAMER = SkyBeamer(sky.target[3], beam_directions=beam_directions)
REDUCER = ift.JaxLinearOperator(
    sky.target,
    SKY_BEAMER.domain,
    lambda x: x[0, 0, 0],  # FIXME: How to do this on all polarizations??
    domain_dtype=tmp_sky.dtype
)


def build_response(field_key, dx, dy, obs, sky_dtype):
    RADIO_RESPONSE = rve.SingleResponse(
        SKY_BEAMER.target[field_key],
        obs.uvw,
        obs.freq,
        do_wgridding=False,
        epsilon=1e-3,
        # center of the dirty image relative to the phase_center
        # (in projected radians)
        center_x=dx,
        center_y=dy,
    )

    FIELD_EXTRACTOR = ift.JaxLinearOperator(
        SKY_BEAMER.target,
        RADIO_RESPONSE.domain,
        lambda x: x[field_key],
        domain_dtype={k: sky_dtype for k, v in SKY_BEAMER.target.items()}
    )

    # FIXME: This is a hack for making stokes I work, see above
    UPCAST_TO_STOKES_I = ift.JaxLinearOperator(
        RADIO_RESPONSE.target,
        obs.vis.domain,
        lambda x: x[None],
        domain_dtype=obs.vis.dtype
    )

    return UPCAST_TO_STOKES_I @ RADIO_RESPONSE @ FIELD_EXTRACTOR


def build_energy(response, obs):
    return rve.DiagonalGaussianLikelihood(
        data=obs.vis,
        inverse_covariance=obs.weight,
        mask=obs.mask
    ) @ response


responses = []
energies = []
for kk, vv, obs in zip(beam_directions.keys(), beam_directions.values(), all_obs):
    R = build_response(kk, vv['dx'], vv['dy'], obs, tmp_sky.dtype)
    responses.append(R @ SKY_BEAMER @ REDUCER)
    energies.append(build_energy(R, obs))


for ii in range(1):
    rnd = ift.from_random(sky.domain)
    f = sky(rnd)

    tot_response_adjoint = np.zeros_like(f.val)
    for rr, oo in zip(responses, all_obs):
        dirty = rr.adjoint(oo.vis).val
        plt.imshow(dirty[0, 0, 0], origin='lower')
        plt.show()
        tot_response_adjoint += dirty
    plt.imshow(tot_response_adjoint[0, 0, 0], origin='lower')
    plt.show()

lh = reduce(lambda x, y: x+y, energies)
lh = lh @ SKY_BEAMER @ REDUCER @ sky


def callback(samples, i):
    sky_mean = samples.average(sky)
    plt.imshow(sky_mean.val[0, 0, 0].T, origin="lower", norm=LogNorm())
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
    export_operator_outputs=dict(
        logdiffuse_stokesI=sky_diffuse_operators['logdiffuse stokesI']),
    resume=True
)

sky_mean = samples.average(sky)
rve.ubik_tools.field2fits(sky_mean, join(
    output_directory, f'sky_reso_{npix}.fits'))