from setuptools import setup, find_packages


# Function to read the contents of the requirements.txt file
def read_requirements():
    with open("requirements.txt") as req:
        return req.read().splitlines()


setup(
    name="ExpandedTM",
    version="0.1.0",
    packages=find_packages(exclude=["examples", "examples.*", "tests", "tests.*"]),
    install_requires=read_requirements(),
    description="A package for expanded topic modeling and metrics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="anton.thielmann@tu-clausthal.de",
    url="https://github.com/yourusername/ExpandedTM",
    python_requires=">=3.6",
)
