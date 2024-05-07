import resolve as rve
import argparse
from os.path import join
from os import makedirs


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


name = input.split('/')[-1] + '_fld{field:02d}'

obs = rve.ms2observations_all(input, "DATA")
for ii, o in enumerate(obs):
    if o is None:
        continue
    o.save(join(outdir, name.format(field=ii)), False)
