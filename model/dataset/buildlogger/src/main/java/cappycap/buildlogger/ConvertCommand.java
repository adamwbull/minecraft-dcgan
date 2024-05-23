package cappycap.buildlogger;

import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;

public class ConvertCommand implements CommandExecutor {

    private BuildLogger plugin;

    public ConvertCommand(BuildLogger plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command cmd, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage("This command can only be run by a player.");
            return true;
        }
        
        Player player = (Player) sender;

        if (!player.hasPermission("buildlogger.convert")) {
            player.sendMessage("You don't have permission to use this command.");
            return true;
        }

        plugin.convertSchematics();
        player.sendMessage("Schematics conversion completed.");
        return true;
    }
}
