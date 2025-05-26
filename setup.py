from setuptools import setup, find_packages

setup(
    name="surface_area_measurement",
    version="0.1.0",
    description="Surface area measurement tools & SAM2 integration",
    author="Ben Cornelisse",
    author_email="b.cornelisse@student.tudelft.nl",
    url="https://github.com/BenCornelisseCode/surface-area-measurement",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0",
        "hydra-core>=1.3",
        # any other runtime requirements, e.g.:
        # "opencv-python",
        # "numpy",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black",
            "flake8",
        ]
    },
    entry_points={
        "console_scripts": [
            # if you want to expose your script as a CLI tool, e.g.
            # "measure=surface_area_measurement.validationSam2Testing:main",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
