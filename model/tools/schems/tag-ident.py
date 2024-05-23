import os
import re

def get_all_tags():
    # path to directory with .schem files
    directory_path = './Schematics'

    # pattern to match your filenames
    # we assume all .schem files follow the format "tag1_tag2_..._tagN.schem"
    pattern = re.compile(r'^(.*).schem$')

    # set to store unique tags
    unique_tags = set()

    # navigate through directory
    for filename in os.listdir(directory_path):
        # if the file is a .schem file
        if filename.endswith('.schem'):
            match = pattern.match(filename)
            if match:
                # split the tags by underscore
                tags = match.group(1).split('_')
                # add tags to the set of unique tags
                unique_tags.update(tags)
    
    return unique_tags

tags = get_all_tags()
print('Unique tags:', tags)
