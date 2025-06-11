import subprocess
import sys
import os


#Function to install the required libraries
def install_libraries():
    libraries = ['matplotlib', 'opencv-python']

    # Install standard libraries
    for lib in libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

    # Install pydensecrf from GitHub
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/lucasb-eyer/pydensecrf.git'])


#Function to run main in given subfolder
def run_main_in_subfolder(subfolder_name):
    folder_path = os.path.join(os.getcwd(), subfolder_name)
    main_file_path = os.path.join(folder_path, 'main.py')
    if os.path.exists(main_file_path):
        print(f"Running {main_file_path}...")
        subprocess.check_call([sys.executable, main_file_path])
    else:
        print(f"main.py not found in {subfolder_name}.")


def main():
    print("Installing required libraries...")
    install_libraries()
    subfolders = ['weakly_supervised_segmentation_baseline',
                  'weakly_supervised_segmentation_improved_model']
    for subfolder in subfolders:
        run_main_in_subfolder(subfolder)


if __name__ == "__main__":
    main()