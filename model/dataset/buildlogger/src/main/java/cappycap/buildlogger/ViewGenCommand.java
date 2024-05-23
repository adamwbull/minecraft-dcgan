// ViewEntryCommand.java

package cappycap.buildlogger;

import com.google.gson.*;
import com.sk89q.worldedit.*;
import com.sk89q.worldedit.bukkit.BukkitAdapter;
import com.sk89q.worldedit.math.BlockVector3;
import com.sk89q.worldedit.session.*;
import com.sk89q.worldedit.world.block.BlockStateHolder;
import me.filoghost.holographicdisplays.api.hologram.Hologram;
import me.filoghost.holographicdisplays.api.HolographicDisplaysAPI;
import org.bukkit.Location;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;

import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class ViewGenCommand implements CommandExecutor {

    private BuildLogger plugin;
    private Map<String, JsonArray> cachedStructures = new HashMap<>(); // Cache for parsed structures

    public ViewGenCommand(BuildLogger plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command cmd, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage("Only players can execute this command.");
            return true;
        }

        Player player = (Player) sender;

        if (args.length == 0) {
            player.sendMessage("Please specify a number or 'list'.");
            return true;
        }

        File directory = new File(plugin.getDataFolder(), "schematics-json-generated");
        File[] files = directory.listFiles();

        if (files == null) {
            player.sendMessage("Error accessing generated entries.");
            return true;
        }

        if (args[0].equalsIgnoreCase("list")) {
            for (int i = 0; i < files.length; i++) {
                player.sendMessage((i + 1) + " - " + files[i].getName());
            }
            return true;
        }

        int index = Integer.parseInt(args[0]);
        if (index < 1 || index > files.length) {
            player.sendMessage("Invalid entry number.");
            return true;
        }

        File selectedFile = files[index - 1];
        JsonArray matrix = null;

        // Check if the structure is cached
        if (cachedStructures.containsKey(selectedFile.getAbsolutePath())) {
            matrix = cachedStructures.get(selectedFile.getAbsolutePath());
        } else {
            try {
                JsonObject jsonObject = JsonParser.parseReader(new FileReader(selectedFile)).getAsJsonObject();
                matrix = jsonObject.getAsJsonArray("matrix");
                cachedStructures.put(selectedFile.getAbsolutePath(), matrix);
            } catch (Exception e) {
                player.sendMessage("Error loading the selected structure.");
                player.sendMessage(e.toString());
                return true;
            }
        }

        // Attempt display.
        try {
            plugin.displayStructure(player, matrix, selectedFile);
        } catch (Exception e) {
            player.sendMessage("Error spinning up the selected structure.");
            player.sendMessage(e.toString());
            return true;
        }

        return true;
    }

}
