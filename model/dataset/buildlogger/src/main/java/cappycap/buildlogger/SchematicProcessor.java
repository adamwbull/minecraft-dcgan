package cappycap.buildlogger;

import com.google.common.collect.ImmutableMap;
import com.google.gson.*;
import com.sk89q.worldedit.extent.clipboard.io.*;
import com.sk89q.worldedit.extent.clipboard.*;
import com.sk89q.worldedit.internal.helper.MCDirections;
import com.sk89q.worldedit.math.BlockVector3;
import com.sk89q.worldedit.regions.Region;
import com.sk89q.worldedit.world.block.BlockState;
import com.sk89q.worldedit.world.block.BlockType;
import com.sk89q.worldedit.world.block.BlockTypes;
import com.sk89q.worldedit.registry.state.Property;
import com.sk89q.worldedit.registry.state.DirectionalProperty;
import com.sk89q.worldedit.util.Direction;
import com.sk89q.worldedit.registry.state.EnumProperty;

import org.bukkit.block.Block;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.Material;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;

public class SchematicProcessor {

    private File pluginDataFolder;

    public SchematicProcessor(File pluginDataFolder) {
        this.pluginDataFolder = pluginDataFolder;
    }

    private static final Map<String, Integer> blockNamesToIds = new HashMap<>();
    private static final Map<String, Integer> directionToBitPosition = new HashMap<>();

    private static final Map<String, Integer> axisToBitPosition = new HashMap<>();

    static {

        // Fill in the block mappings similar to the Python code
        blockNamesToIds.put("minecraft:air", 0);
        blockNamesToIds.put("minecraft:dirt", 1);
        blockNamesToIds.put("minecraft:stone", 2);
        blockNamesToIds.put("minecraft:cobblestone", 3);
        blockNamesToIds.put("minecraft:stone_bricks", 4);
        blockNamesToIds.put("minecraft:oak_slab", 5);
        blockNamesToIds.put("minecraft:oak_planks", 6);
        blockNamesToIds.put("minecraft:oak_log", 7);
        blockNamesToIds.put("minecraft:oak_stairs", 8);
        blockNamesToIds.put("minecraft:glass", 9);
        blockNamesToIds.put("minecraft:white_wool", 10);

        // For converting facing property value to embedded vector index position.
        directionToBitPosition.put("north", 11);
        directionToBitPosition.put("east", 12);
        directionToBitPosition.put("south", 13);
        directionToBitPosition.put("west", 14);

        // For converting axis property value to embedded vector index position.
        axisToBitPosition.put("x", 11);
        axisToBitPosition.put("y", 12);
        axisToBitPosition.put("z", 13);

    }

    // Constructs a list of all possible Minecraft blocks.
    private Set<String> getAllMinecraftBlocks() {
        Set<String> allBlocks = new HashSet<>();
        for (Material material : Material.values()) {
            if (material.isBlock()) {
                allBlocks.add("minecraft:" + material.name().toLowerCase(Locale.ROOT));
            }
        }
        return allBlocks;
    }

    // 
    public void testAllBlocksAndReport() {
        Set<String> allBlocks = getAllMinecraftBlocks();
        Set<String> reportedBlocks = new HashSet<>();
        
        for (String blockName : allBlocks) {
            getBlockIdPosition(blockName, reportedBlocks);
        }

        if (!reportedBlocks.isEmpty()) {
            Bukkit.getLogger().warning("Missing blocks exist in SchematicProcessor.");
        } else {
            Bukkit.getLogger().info("All blocks are handled by SchematicProcessor.");
        }
    }

    public static int getBlockIdPosition(String blockType, Set<String> reportedBlocks) {

        // Decoration blocks.
        Set<String> decorBlocks = Set.of(
                "minecraft:sea_pickle","minecraft:lever",
                "minecraft:chain",
                "minecraft:bell", "minecraft:dripstone_block", "minecraft:peony", "minecraft:moss_block",
                "minecraft:blue_orchid", "minecraft:composter", "minecraft:chest", "minecraft:furnace",
                "minecraft:blast_furnace", "minecraft:azalea", "minecraft:bookshelf", "minecraft:crafting_table",
                "minecraft:smoker", "minecraft:skeleton_skull",
                "minecraft:amethyst_cluster", "minecraft:flowering_azalea", "minecraft:cave_vines", "minecraft:cave_vines_plant",
                "minecraft:spore_blossom", "minecraft:torch", "minecraft:observer",
                "minecraft:hopper", "minecraft:redstone_wire", "minecraft:sticky_piston",
                "minecraft:light_gray_shulker_box", 
                "minecraft:loom", "minecraft:beacon", "minecraft:daylight_detector", "minecraft:chipped_anvil",
                "minecraft:cake", "minecraft:water_cauldron", "minecraft:scaffolding", "minecraft:cartography_table",
                "minecraft:iron_bars",
                "minecraft:target", "minecraft:chiseled_bookshelf",
                "minecraft:structure_block", "minecraft:jigsaw", "minecraft:lightning_rod", "minecraft:damaged_anvil", 
                "minecraft:jukebox", "minecraft:large_amethyst_bud", "minecraft:pink_petals",
                "minecraft:moving_piston", "minecraft:comparator", "minecraft:chain_command_block", 
                "minecraft:anvil", "minecraft:enchanting_table",
                "minecraft:structure_void", "minecraft:repeating_command_block", "minecraft:command_block",
                "minecraft:soul_campfire",  "minecraft:end_rod",
                "minecraft:repeater", "minecraft:dispenser", "minecraft:dropper",
                "minecraft:trapped_chest", "minecraft:powder_snow_cauldron", "minecraft:tnt",
                "minecraft:calibrated_sculk_sensor", "minecraft:conduit", "minecraft:fletching_table",
                "minecraft:lectern", "minecraft:campfire", "minecraft:respawn_anchor",
                "minecraft:note_block", "minecraft:activator_rail", "minecraft:tripwire",
                "minecraft:wall_torch"
        );

        if (decorBlocks.contains(blockType)) {
            blockType = "minecraft:air";
        }
        
        if (blockType.contains("pressure_plate")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("head")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("froglight")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("button")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("carpet")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("door")) {
            blockType = "minecraft:air";
        }

        // Nature blocks.
        Set<String> natureBlocks = Set.of(
            "minecraft:fire", "minecraft:lily_pad", "minecraft:kelp_plant",
            "minecraft:sniffer_egg", "minecraft:sunflower", "minecraft:carrots",
            "minecraft:lava", "minecraft:shroomlight", "minecraft:cobweb", "minecraft:weeping_vines",
            "minecraft:warped_hyphae", "minecraft:pointed_dripstone", "minecraft:small_amethyst_bud", "minecraft:budding_amethyst",
            "minecraft:medium_amethyst_bud", "minecraft:sugar_cane", 
            "minecraft:nether_wart", "minecraft:sculk_shrieker", "minecraft:dragon_egg",
            "minecraft:fern", "minecraft:mangrove_propagule", "minecraft:cave_air", "minecraft:cocoa", "minecraft:bubble_column", "minecraft:nether_sprouts",
            "minecraft:honey_block", "minecraft:pitcher_crop", "minecraft:nether_portal", 
             "minecraft:wheat",
            "minecraft:hay_block", "minecraft:mushroom", "minecraft:brown_mushroom", 
            "minecraft:lily_of_the_valley", "minecraft:kelp", "minecraft:soul_fire", 
            "minecraft:turtle_egg", "minecraft:poppy", "minecraft:big_dripleaf",
            "minecraft:redstone_torch", "minecraft:stonecutter", "minecraft:wither_rose",
            "minecraft:cauldron", "minecraft:oxeye_daisy", "minecraft:wet_sponge",
            "minecraft:sponge", "minecraft:beehive", "minecraft:repeater",
            "minecraft:allium", "minecraft:large_fern", "minecraft:vine",
            "minecraft:sculk_vein", "minecraft:light", "minecraft:soul_torch",
            "minecraft:ladder", "minecraft:warped_nylium", "minecraft:void_air",
            "minecraft:wither_skeleton_skull", "minecraft:frogspawn",
            "minecraft:crimson_fungus","minecraft:slime_block",
            "minecraft:glow_lichen", "minecraft:frosted_ice", "minecraft:azure_bluet",
            "minecraft:powder_snow", "minecraft:melon",
            "minecraft:lilac", "minecraft:jack_o_lantern",
            "minecraft:twisting_vines", "minecraft:sculk_sensor",
            "minecraft:crimson_nylium", 
            "minecraft:bee_nest", "minecraft:sculk_catalyst",
            "minecraft:dried_kelp_block", 
            "minecraft:small_dripleaf", "minecraft:warped_fungus", "minecraft:honeycomb_block",
            "minecraft:spawner", "minecraft:tripwire_hook", "minecraft:powerail",
            "minecraft:lava_cauldron", "minecraft:piston", "minecraft:smithing_table",
            "minecraft:grindstone", "minecraft:brewing_stand", "minecraft:rail",
            "minecraft:barrier", "minecraft:cactus", "minecraft:red_mushroom", "minecraft:dandelion"
        ); 

        if (natureBlocks.contains(blockType)) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("pumpkin")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("roots")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("coral_block")) {
            blockType = "minecraft:stone"; // Not in correct section, but necessary before coral.
        }

        if (blockType.contains("coral")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("stem")) {
            blockType = "minecraft:air";
        }

        // Plants.
        if (blockType.contains("tulip")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("grass") && !blockType.contains("grass_block")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("flower")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("sapling")) {
            blockType = "minecraft:air";
        }

        if (blockType.contains("plant")) {
            blockType = "minecraft:air";
        }

        // Bushes.
        if (blockType.contains("bush")) {
            blockType = "minecraft:air";
        }

        // Candle-related items.
        if (blockType.contains("candle")) {
            blockType = "minecraft:air";
        }

        // Trapdoors.
        if (blockType.contains("trapdoor")) {
            blockType = "minecraft:air";
        }

        // Shulkers.
        if (blockType.contains("shulker_box")) {
            blockType = "minecraft:air";
        }

        // Banners.
        if (blockType.contains("banner")) {
            blockType = "minecraft:air";
        }

        // Signs.
        if (blockType.contains("sign")) {
            blockType = "minecraft:air";
        }

        // Rails.
        if (blockType.contains("rail")) {
            blockType = "minecraft:air";
        }

        // Convert mossy to similar non-mossy type
        if (blockType.contains("mossy_")) {
            blockType = blockType.replace("mossy_", "");
        }

        // Convert wood types to oak.
        Set<String> woods = Set.of("spruce", "birch", "jungle", "acacia", "dark_oak", "cherry");
        for (String wood : woods) {
            if (blockType.contains(wood)) {
                blockType = blockType.replace(wood, "oak");
            }
        }

        // Simplify brick types.
        Set<String> bricks = Set.of(
                "minecraft:polished_diorite",
                "minecraft:granite",
                 "minecraft:quartz_block",
                "minecraft:quartz_pillar", "minecraft:chiseled_quartz_block",
                "minecraft:chiseled_deepslate", "minecraft:polished_deepslate", "minecraft:cracked_deepslate_tiles",
                "minecraft:polished_andesite", "minecraft:amethyst_block", "minecraft:oxidized_cut_copper",
                "minecraft:raw_copper_block", "minecraft:emerald_block",
                "minecraft:coal_block", "minecraft:waxed_exposed_copper", "minecraft:gold_block", "minecraft:oxidized_copper",
                "minecraft:crimson_hyphae"
        );

        if (bricks.contains(blockType)) {
            blockType = "minecraft:stone_bricks";
        }

        if (blockType.contains("brick")) {
            blockType = "minecraft:stone_bricks";
        }  

        // Ground-like blocks into dirt.
        Set<String> groundBlocks = Set.of(
                "minecraft:redstone_block",
                "minecraft:mycelium", "minecraft:podzol", "minecraft:grass_block",
                "minecraft:sand", "minecraft:gravel", "minecraft:clay", "minecraft:snow",
                "minecraft:snow_block", "minecraft:packed_ice", "minecraft:blue_ice", "minecraft:ice",
                "minecraft:coarse_dirt", "minecraft:farmland", "minecraft:rooted_dirt", "minecraft:nether_wart_block",
                "minecraft:soul_sand", "minecraft:packed_mud", "minecraft:dirt_path", "minecraft:netherrack",
                "minecraft:mud", "minecraft:suspicious_sand", "minecraft:magma_block", "minecraft:ancient_debris",
                "minecraft:smooth_quartz", "minecraft:red_sand", "minecraft:moss_block", "minecraft:soul_soil", "minecraft:frosted_ice",
                "minecraft:water"
        );

        if (groundBlocks.contains(blockType)) {
            blockType = "minecraft:dirt";
        }

        // Cobble types.
        Set<String> cobbleTypes = Set.of(
                "minecraft:cobbled_deepslate", "minecraft:deepslate", "minecraft:infested_cobblestone"
        );

        if (cobbleTypes.contains(blockType)) {
            blockType = "minecraft:cobblestone";
        }

        // Stone types.
        Set<String> stoneTypes = Set.of(
                "minecraft:glowstone",
                "minecraft:andesite", "minecraft:diorite", "minecraft:sandstone", "minecraft:obsidian",
                "minecraft:sculk", "minecraft:fire_coral_block", "minecraft:dead_bubble_coral_block",
                "minecraft:tuff", "minecraft:suspicious_gravel", "minecraft:infested_stone",
                "minecraft:purpur_block", "minecraft:blackstone", "minecraft:lodestone", "minecraft:purpur_pillar",
                "minecraft:gilded_blackstone",
                "minecraft:polished_blackstone", "minecraft:raw_gold_block", 
                "minecraft:crying_obsidian","minecraft:chiseled_polished_blackstone", "minecraft:raw_iron_block", 
                "minecraft:iron_block", "minecraft:lapis_block", "minecraft:diamond_block",
                "minecraft:end_stone", "minecraft:prismarine",
                "minecraft:smooth_stone", "minecraft:polished_granite", "minecraft:calcite", 
                "minecraft:netherite_block", "minecraft:warped_wart_block", "minecraft:dark_prismarine",
                "minecraft:lantern", "minecraft:redstone_lamp", "minecraft:sea_lantern", "minecraft:soul_lantern"
        );

        if (stoneTypes.contains(blockType)) {
            blockType = "minecraft:stone";
        }

        if (blockType.contains("concrete")) {
            blockType = "minecraft:stone";
        }

        if (blockType.contains("copper")) {
            blockType = "minecraft:stone";
        }

        if (blockType.contains("sandstone")) {
            blockType = "minecraft:stone";
        }

        if (blockType.contains("terracotta")) {
            blockType = "minecraft:stone";
        }

        if (blockType.contains("basalt")) {
            blockType = "minecraft:stone";
        }

        if (blockType.contains("deepslate")) {
            blockType = "minecraft:stone";
        }

        if (blockType.contains("ore")) {
            blockType = "minecraft:stone";
        }

        // Convert double stacked slabs to oak block.
        if (blockType.contains("slab") && blockType.contains("type=double")) {
            blockType = "minecraft:oak_planks";
        }

        // Simplify all other slabs.
        if (blockType.contains("slab")) {
            blockType = "minecraft:oak_slab";
        }

        // Handle bamboo.
        if (blockType.contains("bamboo_block") || blockType.contains("bamboo_planks") || blockType.contains("bamboo_mosaic")) {
            blockType = "minecraft:oak_planks";
        } else if (blockType.contains("bamboo")) {
            blockType = "minecraft:air";
        }

        // Planks catch-all.
        if (blockType.contains("planks")) {
            blockType = "minecraft:oak_planks";
        }

        // Simplify walls.
        if (blockType.contains("wall")) {
            blockType = "minecraft:cobblestone";
        }

        // Simplify stairs.
        if (blockType.contains("stairs")) {
            blockType = "minecraft:oak_stairs";
        }

        // Simple glass blocks only.
        if (blockType.contains("glass")) {
            blockType = "minecraft:glass";
        }

        // Convert wool types to white wool.
        Set<String> decorTypes = Set.of(
                "minecraft:bone_block"
        );

        if (blockType.contains("wool") || decorTypes.contains(blockType)) {
            blockType = "minecraft:white_wool";
        }

        // Stripped and barrel to log.
        if (blockType.contains("stripped") || blockType.contains("minecraft:barrel")) {
            blockType = "minecraft:oak_log";
        }

        if (blockType.contains("log") || blockType.contains("mushroom_block") || blockType.contains("mushroom_stem")) {
            blockType = "minecraft:oak_log";
        }

        if (blockType.contains("_wood")) {
            blockType = "minecraft:oak_log";
        }

        // Fences.
        if (blockType.contains("fence")) {
            blockType = "minecraft:oak_log";
        }

        // Beds.
        if (blockType.contains("bed")) {
            blockType = "minecraft:air";
        }

        // Leaves.
        if (blockType.contains("leaves")) {
            blockType = "minecraft:air";
        }

        // Potted plants.
        if (blockType.contains("pot")) {
            blockType = "minecraft:air";
        }

        // End related blocks that arent endstone.
        if (blockType.contains("end")) {
            blockType = "minecraft:air";
        }

        Integer result = blockNamesToIds.get(blockType);
        if(result == null) {
            if (!reportedBlocks.contains(blockType)) {
                Bukkit.getLogger().warning("Unknown block: " + blockType + ". Using default block.");
                reportedBlocks.add(blockType);
            }
            return 0;
        }
        return result;
    }

    public static int[] blockStateToVector(BlockState blockState, Set<String> reportedBlocks) {

        int[] blockVector = new int[16];
        String blockName = blockState.getBlockType().getId();

        int blockIdPosition = getBlockIdPosition(blockName, reportedBlocks);
        blockVector[blockIdPosition] = 1;

        // Handle direction for blocks with that property.
        String blockString = blockState.toString();

        if (blockString.contains("[")) {

            String properties = blockString.split("\\[")[1].split("\\]")[0];

            for (String property : properties.split(",")) {

                String[] keyValuePair = property.split("=");

                if (keyValuePair.length == 2) {

                    String key = keyValuePair[0].trim();  // trim to remove any whitespace
                    String value = keyValuePair[1].trim(); // trim to remove any whitespace

                    // Handle 'facing' property
                    if (key.equals("facing") && directionToBitPosition.containsKey(value)) {
                        blockVector[directionToBitPosition.get(value)] = 1;
                    }

                    // Handle 'axis' property for logs
                    if (key.equals("axis") && axisToBitPosition.containsKey(value)) {
                        blockVector[axisToBitPosition.get(value)] = 1;
                    }

                    // Handle 'type' and 'half' properties
                    if ((key.equals("type") || key.equals("half")) && value.equals("top")) {
                        blockVector[15] = 1;
                    }

                }

            }

        }

        return blockVector;

    }

    public static BlockState vectorToBlockState(int[] blockVector) {

        if (blockVector.length != 16) {
            throw new IllegalArgumentException("Invalid block vector length.");
        }

        int pos = findPosition(blockVector);
        String blockName = findBlockNameByPosition(pos);

        if (blockName == null) {
            throw new IllegalArgumentException("Invalid block vector.");
        }

        BlockType blockType = BlockTypes.get(blockName);
        if (blockType == null) {
            throw new IllegalArgumentException("No block type found for: " + blockName);
        }

        BlockState finalBlockState = blockType.getDefaultState();

        // oak_stairs directionality property
        if (pos == 8) {

            Property halfProperty = blockType.getProperty("half");
            Property facingProperty = blockType.getProperty("facing");

            String halfPropertyValueString = blockVector[15] == 1 ? "top" : "bottom";
            String facingPropertyValueString = findMatchingKey(blockVector, directionToBitPosition);

            // Convert the string value to the actual Direction enum object
            Direction facingDirection = (Direction) facingProperty.getValueFor(facingPropertyValueString.toUpperCase(Locale.ROOT));

            // Now you set the block state with the properties
            finalBlockState = finalBlockState.with(halfProperty, halfPropertyValueString);
            debugToFile("Before: " + finalBlockState.toString());
            finalBlockState = finalBlockState.with(facingProperty, facingDirection);
            debugToFile("After: " + finalBlockState.toString());

            debugToFile(Arrays.toString(blockVector) + ": " + facingProperty + " " + halfProperty);
            debugToFile(Arrays.toString(blockVector) + ": " + facingDirection + " " + halfPropertyValueString);

        }


        // oak_slab type property
        if (pos == 5) {
            String typeProperty = blockVector[15] == 1 ? "top" : "bottom";
            finalBlockState = finalBlockState.with(blockType.getProperty("type"), typeProperty);
        }

        // oak_log axis property
        if (pos == 7) {
            String axis = findMatchingKey(blockVector, axisToBitPosition);
            if (axis != null) {
                finalBlockState = finalBlockState.with(blockType.getProperty("axis"), axis);
            }
        }

        return finalBlockState;
    }

    private static void debugToFile(String message) {
        try {
            Files.write(Paths.get("debug_log.txt"), message.concat("\n").getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private static int findPosition(int[] blockVector) {
        for (int i = 0; i < blockVector.length; i++) {
            if (blockVector[i] == 1) {
                return i;
            }
        }
        return -1;
    }

    private static String findBlockNameByPosition(int pos) {
        for (Map.Entry<String, Integer> entry : blockNamesToIds.entrySet()) {
            if (entry.getValue().equals(pos)) {
                return entry.getKey();
            }
        }
        return null;
    }

    private static String findMatchingKey(int[] blockVector, Map<String, Integer> map) {
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            if (blockVector[entry.getValue()] == 1) {
                return entry.getKey();
            }
        }
        return null;
    }

    public void processDirectory() {

        Bukkit.getLogger().info("Processing ./schematics directory...");
        File directory = new File(pluginDataFolder, "schematics");

        if (!directory.exists() || !directory.isDirectory()) {
            Bukkit.getLogger().warning("./schematics is not a directory or doesn't exist.");
            return;
        }

        File[] files = directory.listFiles();

        if (files == null) {
            Bukkit.getLogger().warning("Failed to list files in ./schematics directory.");
            return;
        }

        int convertedCount = 0;
        for (File file : files) {
            if (file.getName().endsWith(".schem") || file.getName().endsWith(".schematic") || file.getName().endsWith(".nbt")) {
                processSchematic(file);
                convertedCount++;
            }
        }

        Bukkit.getLogger().info("Schematic conversions complete. Converted schems: " + convertedCount);
    }

    private static JsonArray intArrayToJsonArray(int[] arr) {
        JsonArray jsonArray = new JsonArray();
        for (int val : arr) {
            jsonArray.add(val);
        }
        return jsonArray;
    }

    public void processSchematic(File file) {

        Set<String> reportedBlocks = new HashSet<>();

        ClipboardFormat format = ClipboardFormats.findByFile(file);
        if (format == null) {
            Bukkit.getLogger().warning("Unknown or unsupported schematic format for file: " + file.getName());
            return;
        }

        try (ClipboardReader reader = format.getReader(Files.newInputStream(file.toPath()))) {
            Clipboard clipboard = reader.read();

            // Initialize the 3D matrix
            Region region = clipboard.getRegion();
            int width = region.getWidth();
            int height = region.getHeight();
            int length = region.getLength();

            JsonArray matrix = new JsonArray();
            for (int x = 0; x < width; x++) {
                JsonArray xArray = new JsonArray();
                for (int y = 0; y < height; y++) {
                    JsonArray yArray = new JsonArray();
                    for (int z = 0; z < length; z++) {
                        yArray.add(""); // Initialize empty
                    }
                    xArray.add(yArray);
                }
                matrix.add(xArray);
            }

            // Iterate over all blocks in the clipboard and process them
            for (BlockVector3 blockVector3 : region) {

                BlockState blockState = clipboard.getBlock(blockVector3);

                int[] blockVector = blockStateToVector(blockState, reportedBlocks);
                JsonArray blockVectorJson = intArrayToJsonArray(blockVector);

                // Update the 3D matrix at the block's position
                BlockVector3 min = region.getMinimumPoint();

                int adjustedX = blockVector3.getX() - min.getX();
                int adjustedY = blockVector3.getY() - min.getY();
                int adjustedZ = blockVector3.getZ() - min.getZ();

                JsonArray xArray = matrix.get(adjustedX).getAsJsonArray();
                JsonArray yArray = xArray.get(adjustedY).getAsJsonArray();
                yArray.set(adjustedZ, blockVectorJson);

            }

            JsonObject result = new JsonObject();
            result.add("matrix", matrix);

            // Save the result to a schematics-json folder
            File outputDirectory = new File(pluginDataFolder, "schematics-json");
            if (!outputDirectory.exists()) {
                outputDirectory.mkdir();
            }

            Files.write(Paths.get(outputDirectory + "/" + file.getName() + ".json"), new Gson().toJson(result).getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}