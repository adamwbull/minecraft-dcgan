// UpdateEntryCommand.java

package cappycap.buildlogger;

import com.google.gson.*;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class UpdateEntryCommand implements CommandExecutor {

    private BuildLogger plugin;

    public UpdateEntryCommand(BuildLogger plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command cmd, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage("Only players can execute this command.");
            return true;
        }

        Player player = (Player) sender;

        if (args.length < 2) {
            player.sendMessage("Please specify an entry number and a category.");
            return true;
        }

        int entryNumber;
        try {
            entryNumber = Integer.parseInt(args[0]);
        } catch (NumberFormatException e) {
            player.sendMessage("Invalid entry number format.");
            return true;
        }

        String category = args[1];
        File directory = new File(plugin.getDataFolder(), "schematics-json-preprocessed");
        File[] files = directory.listFiles();

        if (files == null || entryNumber < 1 || entryNumber > files.length) {
            player.sendMessage("Invalid entry number or error accessing schematic entries.");
            return true;
        }

        File selectedFile = files[entryNumber - 1];
        File newCategoryFolder = new File(plugin.getDataFolder(), "schematics-json-preprocessed-" + category);

        if (!newCategoryFolder.exists() && !newCategoryFolder.mkdir()) {
            player.sendMessage("Error creating category folder.");
            return true;
        }

        File newFile = new File(newCategoryFolder, selectedFile.getName());
        try {
            // Remove from previous category if it exists
            removeFileFromOtherCategories(selectedFile, category);
            // Copy to new category
            Gson gson = new Gson();
            JsonObject structureData = gson.fromJson(new FileReader(selectedFile), JsonObject.class);
            try (FileWriter writer = new FileWriter(newFile)) {
                gson.toJson(structureData, writer);
            }

            player.sendMessage("Entry updated and categorized under '" + category + "'.");
        } catch (IOException e) {
            player.sendMessage("Error processing the file.");
            player.sendMessage(e.toString());
            return true;
        }

        return true;
    }

    private void removeFileFromOtherCategories(File file, String currentCategory) throws IOException {
        File parentDir = file.getParentFile().getParentFile();
        File[] categoryDirs = parentDir.listFiles((dir, name) -> name.startsWith("schematics-json-preprocessed-") && !name.endsWith(currentCategory));

        if (categoryDirs != null) {
            for (File dir : categoryDirs) {
                File existingFile = new File(dir, file.getName());
                if (existingFile.exists()) {
                    if (!existingFile.delete()) {
                        throw new IOException("Failed to delete existing file in other category: " + existingFile.getPath());
                    }
                }
            }
        }
    }

}