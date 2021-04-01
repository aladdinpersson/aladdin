from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="aladdin",  # How you named your package folder (MyLib)
    version="0.0.3",  # Start with a small number and increase it with every change you make
    license="MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="Useful deep learning functions",  # Give a short description about your library
    author="Aladdin Persson",  # Type in your name
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    url="https://github.com/aladdinpersson/DL-Utils",
    author_email="aladdin.persson@hotmail.com",  # Type in your E-Mail
    install_requires=[  # I get to this in a second
        "torch",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
