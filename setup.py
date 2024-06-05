import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="nnunetv2",
        packages=setuptools.find_packages(exclude=["docker"]),
    )
