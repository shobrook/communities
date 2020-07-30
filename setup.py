import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from codecs import open

if sys.version_info[:3] < (3, 0, 0):
    print("Requires Python 3 to run.")
    sys.exit(1)

with open("README.md", encoding="utf-8") as file:
    readme = file.read()

setup(
    name="communities",
    description="Library of algorithms for detecting communities in graphs",
    long_description=readme,
    long_description_content_type="text/markdown",
    version="v2.0.0",
    packages=["communities", "communities.algorithms"],
    python_requires=">=3",
    url="https://github.com/shobrook/communities",
    author="shobrook",
    author_email="shobrookj@gmail.com",
    # classifiers=[],
    install_requires=["networkx"],
    keywords=["graph", "louvain", "community", "clustering", "detection", "girvan-newman", "hierarchical"],
    license="MIT"
)