import subprocess
import sys



def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("numpy")
install("tensorflow")
install("keras-rl")
install("gym")
install("gym[atari]")