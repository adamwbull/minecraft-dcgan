package cappycap.buildlogger;

import com.google.gson.JsonArray;
import com.sk89q.worldedit.MaxChangedBlocksException;
import com.sk89q.worldedit.WorldEditException;
import com.sk89q.worldedit.function.operation.Operation;
import com.sk89q.worldedit.session.ClipboardHolder;
import org.bukkit.entity.Player;
import net.md_5.bungee.api.ChatColor;
import net.md_5.bungee.api.chat.TextComponent;
import org.bukkit.plugin.java.JavaPlugin;
import com.sk89q.worldedit.extent.clipboard.BlockArrayClipboard;
import com.sk89q.worldedit.math.BlockVector3;
import com.sk89q.worldedit.regions.CuboidRegion;
import com.google.gson.Gson;
import org.bukkit.Material;
import com.sk89q.worldedit.WorldEdit;
import com.sk89q.worldedit.EditSession;
import com.sk89q.worldedit.function.operation.ForwardExtentCopy;
import com.sk89q.worldedit.function.operation.Operations;
import com.sk89q.worldedit.bukkit.BukkitAdapter;
import org.bukkit.Location;
import com.sk89q.worldedit.world.block.BlockStateHolder;
import java.io.FileInputStream;

import com.sk89q.worldedit.extent.clipboard.Clipboard;
import com.sk89q.worldedit.extent.clipboard.io.ClipboardFormats;
import com.sk89q.worldedit.extent.clipboard.io.ClipboardFormat;
import com.sk89q.worldedit.extent.clipboard.io.ClipboardReader;

import org.bukkit.Bukkit;

import java.io.File;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

public class BuildLogger extends JavaPlugin {

    private SchematicProcessor processor;

    static public TextComponent border = new TextComponent("------------------------------------------------");

    static {
        border.setColor(ChatColor.GRAY);
    }

    // Set up json folders.
    private void createDirectoryStructure() {
        File pluginDirectory = this.getDataFolder();

        if (!pluginDirectory.exists()) {
            pluginDirectory.mkdir();
        }

        File subDirectory = new File(pluginDirectory, "schematics");
        if (!subDirectory.exists()) {
            subDirectory.mkdir();
        }

        File subDirectoryJson = new File(pluginDirectory, "schematics-json");
        if (!subDirectoryJson.exists()) {
            subDirectoryJson.mkdir();
        }

        File subDirectoryJsonPreprocessed = new File(pluginDirectory, "schematics-json-preprocessed");
        if (!subDirectoryJsonPreprocessed.exists()) {
            subDirectoryJsonPreprocessed.mkdir();
        }

        File subDirectoryGenerated = new File(pluginDirectory, "schematics-json-generated");
        if (!subDirectoryGenerated.exists()) {
            subDirectoryGenerated.mkdir();
        }

    }

    @Override
    public void onEnable() {

        createDirectoryStructure();

        getLogger().info("BuildLogger v"+this.getDescription().getVersion()+" initialized.");

        // Register commands.
        this.getCommand("convert").setExecutor(new ConvertCommand(this));
        this.getCommand("ve").setExecutor(new ViewEntryCommand(this));
        this.getCommand("ue").setExecutor(new UpdateEntryCommand(this));
        this.getCommand("vg").setExecutor(new ViewGenCommand(this));

        // Init our SchematicProcessor.
        processor = new SchematicProcessor(this.getDataFolder());

        // Perform block conversion checks and report to console.
        processor.testAllBlocksAndReport();

    }

    // Run our SchematicProcessor's conversion process.
    public void convertSchematics() {
        processor.processDirectory();
    }    

    @Override
    public void onDisable() {
        getLogger().info("BuildLogger disabled. Bye!!!");
    }

    // Display a JsonArray matrix.
    public void displayStructure(Player player, JsonArray matrix, File selectedFile) throws MaxChangedBlocksException {

        if (matrix == null) {
            player.sendMessage("Error: Invalid structure.");
            return;
        }

        BlockVector3 bottomCorner = BlockVector3.at(0, 0, 0);
        BlockVector3 topCorner = BlockVector3.at(matrix.size() - 1, matrix.get(0).getAsJsonArray().size() - 1, matrix.get(0).getAsJsonArray().get(0).getAsJsonArray().size() - 1);
        CuboidRegion region = new CuboidRegion(bottomCorner, topCorner);

        // Create a clipboard to hold the block changes
        BlockArrayClipboard clipboard = new BlockArrayClipboard(region);

        player.sendMessage("Displaying structure...");

        for (int x = 0; x < matrix.size(); x++) {
            JsonArray xArray = matrix.get(x).getAsJsonArray();
            for (int y = 0; y < xArray.size(); y++) {
                JsonArray yArray = xArray.get(y).getAsJsonArray();
                for (int z = 0; z < yArray.size(); z++) {
                    JsonArray jsonArray = yArray.get(z).getAsJsonArray();
                    int[] blockVector = new int[jsonArray.size()];
                    for (int i = 0; i < jsonArray.size(); i++) {
                        blockVector[i] = jsonArray.get(i).getAsInt();
                    }
                    BlockStateHolder block = processor.vectorToBlockState(blockVector);
                    clipboard.setBlock(BlockVector3.at(x, y, z), block);
                }
            }
        }

        try (EditSession session = WorldEdit.getInstance().newEditSession(BukkitAdapter.adapt(player.getWorld()))) {

            // Paste the clipboard content to the world
            Operation operation = new ClipboardHolder(clipboard).createPaste(session).to(bottomCorner).ignoreAirBlocks(false).build();
            Operations.complete(operation);

            session.flushSession();
            player.sendMessage("Structure placed!");
        } catch (WorldEditException e) {
            player.sendMessage("Error displaying the selected structure.");
            player.sendMessage(e.toString());
        }

    }

    public void displaySchemStructure(Player player, File schemFile, File jsonFile) throws Exception {
        
        player.sendMessage("Displaying .schem structure...");

        // Calculate the offset for the .schem structure
        BlockVector3 schemOffset = BlockVector3.at(-65, 0, 0);

        Clipboard clipboard;

        ClipboardFormat format = ClipboardFormats.findByFile(schemFile);
        try (ClipboardReader reader = format.getReader(new FileInputStream(schemFile))) {

            clipboard = reader.read();

            try (EditSession editSession = WorldEdit.getInstance().newEditSession(BukkitAdapter.adapt(player.getWorld()))) {
                Operation operation = new ClipboardHolder(clipboard)
                        .createPaste(editSession)
                        .to(schemOffset)
                        .build();
                Operations.complete(operation);

            } catch (Exception e) {
                throw new Exception("Error building .schem structure: " + e.getMessage(), e);
            }

        } catch (Exception e) {
            throw new Exception("Error loading .schem structure: " + e.getMessage(), e);
        }

        player.sendMessage("Schem structure placed!");

    }

}
