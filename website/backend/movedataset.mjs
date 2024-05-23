import fs from 'fs';
import path from 'path';
import schematic2schemModule from 'schematic2schem';

const schematic2schem = schematic2schemModule.default;

// Define source and destination folders
const sourceFolder = 'C:\\Users\\A\\Projects\\minecraft-GAN\\dataset\\server\\plugins\\BuildLogger\\schematics';
const destinationFolder = 'C:\\Users\\A\\Projects\\minecraft-gan-website\\backend\\py\\schematics\\dataset';
const convertedFolder = path.join(sourceFolder, 'converted'); // Subfolder for already converted files

// Ensure the converted folder exists
fs.mkdir(convertedFolder, { recursive: true }, (err) => {
  if (err) {
    console.error("Error creating/verifying the converted folder:", err);
    return;
  }

  // Read the source directory for .schematic and .schem files
  fs.readdir(sourceFolder, (err, files) => {
    if (err) {
      console.error("Error reading source directory:", err);
      return;
    }

    files.forEach(file => {
      const sourcePath = path.join(sourceFolder, file);
      const destinationPath = path.join(destinationFolder, file);
      const convertedPath = path.join(convertedFolder, file);

      // Check if the file is a .schematic or .schem file
      if (file.endsWith('.schematic')) {
        // Determine if the .schem version already exists in the destination folder
        const schemFileName = file.replace('.schematic', '.schem');
        const schemFilePath = path.join(destinationFolder, schemFileName);

        fs.access(schemFilePath, fs.constants.F_OK, (err) => {
          if (!err) {
            // .schem file exists, move .schematic to /converted
            fs.rename(sourcePath, convertedPath, (err) => {
              if (err) {
                console.log(`Error moving already converted file ${file} to ${convertedPath}:`, err);
              } else {
                console.log(`${file} has already been converted and was moved to ${convertedPath}`);
              }
            });
          } else {
            // .schem file does not exist, proceed with conversion
            fs.readFile(sourcePath, async (err, data) => {
              if (err) {
                console.log(`Error reading file ${file}:`, err);
                return;
              }

              try {
                const schemBuffer = await schematic2schem(data);
                const newDestinationPath = schemFilePath; // Use schemFilePath which already has the .schem extension
                fs.writeFile(newDestinationPath, schemBuffer, (err) => {
                  if (err) {
                    console.log(`Error writing file ${newDestinationPath}:`, err);
                  } else {
                    console.log(`${file} converted and moved to ${newDestinationPath}`);
                    // Optionally delete or move the original .schematic file after successful conversion
                    // fs.unlink(sourcePath, (err) => {
                    //   if (err) console.log(`Error deleting original file ${sourcePath}:`, err);
                    // });
                  }
                });
              } catch (err) {
                console.log(`Error converting file ${file}:`, err);
              }
            });
          }
        });
      } else if (file.endsWith('.schem')) {
        // Move .schem files directly without conversion
        fs.rename(sourcePath, destinationPath, (err) => {
          if (err) {
            console.log(`Error moving file ${file}:`, err);
          } else {
            console.log(`${file} moved to ${destinationPath}`);
          }
        });
      }
    });
  });
});
