import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
        name="pvi",
        version="0.0.1",
        author="Matt Ashman",
        author_email="mca39@cam.ac.uk",
        description="Partitioned variational inference",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/MattAshman/pvi",
        project_urls={
            "Bug Tracker": "https://github.com/MattAshman/pvi/issues",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        package_dir={"": "."},
        packages=setuptools.find_packages(where="."),
        python_requires=">=3.6",
)
