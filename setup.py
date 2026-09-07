from setuptools import setup, find_packages

setup(
    name="oste-moe",
    version="1.0.0",
    description="Hierarchical Multi-Modal Mixture-of-Experts Exoplanet Discovery & Characterization Engine",
    author="OSTE-MoE Research Consortium",
    author_email="research@oste-moe.org",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "astropy>=5.0",
        "scipy>=1.10.0",
        "lightkurve>=2.4.0"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)
