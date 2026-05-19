import os

from config import PROJECT_DIR, CIFAR10C_DIR


def main():
    os.makedirs(PROJECT_DIR, exist_ok=True)

    if not os.path.exists(CIFAR10C_DIR):
        print("Downloading CIFAR-10-C...")
        os.system("wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1")
        print("Extracting CIFAR-10-C...")
        os.system("tar -xf CIFAR-10-C.tar")
        print("Done.")
    else:
        print("CIFAR-10-C already exists.")

    os.system("ls CIFAR-10-C | head")


if __name__ == "__main__":
    main()
