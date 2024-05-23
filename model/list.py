import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):

        # Skip any folders.
        valid_root = True
        invalid_roots = [
            'tools',
            'wrappers'
        ]

        for root_substring in invalid_roots:
            if root_substring in root:
                valid_root = False
                break  

        # Filter out the files with extensions we are interested in
        relevant_files = [file for file in files if file.endswith(('.py', '.java'))]
        
        if relevant_files and valid_root:
            print(f'{root} contains:')
            for file in relevant_files:
                print(file)
            print('')  # Print a newline for better readability between directories

if __name__ == '__main__':
    list_files('.')
