import resolve as rve
import argparse
from os.path import join
from os import makedirs

import numpy as np

from sys import exit

description = 'Extracts interferometer data from a measurement set.'
parser = argparse.ArgumentParser(prog='extract_data', description=description)
parser.add_argument('input_file', help='path to measurement set')
parser.add_argument(
    "-o",
    "--output-dir",
    default='.',
    help="output directory, default: .",
    metavar='<dir>')
args = parser.parse_args()
input = args.input_file
outdir = args.output_dir
makedirs(outdir, exist_ok=True)


spw_length = 3  # FIXME: SHOULD be set dynamically
fields = [
    'A523_TL', 'A523_TR', 'A523_ML', 'A523_MC', 'A523_MR', 'A523_BL']  # FIXME: SHOULD be set dynamically
field_ids = [ii for ii in range(5, 11)]


name = input.split('/')[-1] + '_fld{field:02d}_spw{spw:02d}'


observations = {}
for field_id, field in zip(field_ids, fields):
    for spw in range(spw_length):

        o = rve.ms2observations(
            ms=input,
            data_column="DATA",
            with_calib_info=False,
            spectral_window=spw,
            field=field)

        for oo in o:
            if oo is not None:
                oo.save(join(outdir, name.format(field=field_id, spw=spw)), False)
