from setuptools import setup, find_packages

if __name__ == '__main__':
    # __version__ = get_version()

    setup(
        name='sdidc-rl',
        version="0.0.1",
        include_package_data=True,
        packages=find_packages(),
        install_requires=[
            "numpy",
            "matplotlib"
        ]
    )
