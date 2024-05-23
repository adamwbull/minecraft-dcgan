import mcschematic

schem = mcschematic.MCSchematic()

schem.setBlock((0, 0, 0), "minecraft:oak_log[axis=z]")
schem.setBlock((0, 1, 0), "minecraft:oak_slab[type=top,waterlogged=false]")
schem.setBlock((0, 2, 0), "minecraft:oak_stairs[facing=west,half=bottom,shape=straight,waterlogged=false]")
schem.setBlock((0, 3, 0), "minecraft:oak_log[axis=y]")

schem.save(outputFolderPath='./schematics', schemName='kakunasugh', version=mcschematic.Version.JE_1_18)