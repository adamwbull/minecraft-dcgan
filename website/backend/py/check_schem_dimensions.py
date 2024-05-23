import argparse
from mcschematic import MCSchematic

def check_dimensions(schem_path):
    schem = MCSchematic(schematicToLoadPath_or_mcStructure=schem_path)
    for x in range(32):
        for y in range(32):
            for z in range(32):
                if schem.getBlockDataAt((x, y, z)) is None:
                    return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check .schem file dimensions")
    parser.add_argument("--schem_path", type=str, required=True, help="Path to the schematic file")

    args = parser.parse_args()

    if check_dimensions(args.schem_path):
        print("OK")
    else:
        print("Dimensions out of bounds")
