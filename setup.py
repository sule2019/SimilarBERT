import pathlib
import setuptools

# Include your long_description content or provide it as a variable (e.g., long_description)
long_description = "Documentation will update soon"

# Define the required packages (replace with your actual dependencies)
base_packages = []

setuptools.setup(
    name="SimilarBERT",
    packages=setuptools.find_packages(),
    version="0.1.0",
    author="Mohammed Sule",
    author_email="mohammed.sule922015@gmail.com",
    license="MIT License",  # Add a comma to separate license from the next field
    description="SimilarBERT uses fast clustering with BERT to perform topic modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sule2019/SimilarBERT",
    project_urls={
        "Documentation": "https://github.com/sule2019/SimilarBERT",
        "Source Code": "https://github.com/sule2019/SimilarBERT",
        "Issue Tracker": "https://github.com/sule2019/SimilarBERT/issues",
    },
    keywords="nlp bert topic modeling embeddings",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=base_packages,
    include_package_data=True, 
    python_requires='>=3.7',
)
