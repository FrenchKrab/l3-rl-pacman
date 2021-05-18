import subprocess
import sys


def get_package_path(package):
    return "./envs/" + get_package_name(package)

def get_package_name(package):
    return "gym-"+package

def uninstall(package):
    subprocess.check_call([sys.executable, "-m", "pip",  "uninstall", "-y", get_package_name(package)])

def install_local(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", get_package_path(package)])



packages = ["tresor2d"]

for p in packages:
    install_local(p)