# gen_single.py
import torch
import argparse
from generators.buildgenerator import BuildGenerator
from variables.globals import default_height, default_width, default_depth

parser = argparse.ArgumentParser(description='Generate a single image using cDCGAN.')
parser.add_argument('--height', type=int, default=default_height, help='Height of the generated image.')
parser.add_argument('--width', type=int, default=default_width, help='Width of the generated image.')
parser.add_argument('--depth', type=int, default=default_depth, help='Depth of the generated image.')
parser.add_argument('--generator', type=str, required=True, help='Path to the generator checkpoint.')
parser.add_argument('--prompt', type=str, required=True, help='Text prompt for the generator.')

args = parser.parse_args()

# Load generator
generator = BuildGenerator(args.generator)

generator.generate_build(args.height, args.width, args.depth, args.prompt)

# Display or save the result later...
