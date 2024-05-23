import os
import re

def clean_filenames():
    # path to directory with .schem files
    directory_path = './Schematics'

    # navigate through directory
    for filename in os.listdir(directory_path):
        # if the file is a .schem file
        if filename.endswith('.schem'):
            # Split the filename into name and extension
            name, ext = os.path.splitext(filename)

            # replace undesired labels in filename
            name = re.sub(r'(one|two|three)_(floors?|stories?)', r'\1floor', name)
            name = re.sub(r'_with_|_very_|_building_|very_', '', name)

            # New complete filename with cleaned name
            new_filename = f'{name}{ext}'

            # rename the file
            os.rename(
                os.path.join(directory_path, filename),
                os.path.join(directory_path, new_filename)
            )

clean_filenames()
