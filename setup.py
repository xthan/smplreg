from setuptools import setup, find_packages

setup(
    name="smplreg",
    packages=find_packages(),
    version="0.1",
    license="MIT",
    description="Registration between a reconstructed point cloud and an estimated SMPL mesh",
    author="Xintong Han",
    author_email="hixintonghan@gmail.com",
    url="https://github.com/xthan/smplreg",
    keywords=["digital human", "SMPL", "body capture", "deep learning"],
    install_requires=[
        "torch",
        "pytorch3d",
        "smplxd@git+https://github.com/xthan/smplxd.git",
        "omegaconf",
        "chumpy",
        "pre-commit",
        "black==20.8b1",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
