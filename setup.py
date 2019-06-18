from setuptools import setup, find_packages

# from https://github.com/justusschock/template-repo-python/blob/master/setup.py
import os
import re


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


def find_version(file):
    content = read_file(file)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content,
                              re.M)
    if version_match:
        return version_match.group(1)


requirements = resolve_requirements(os.path.join(os.path.dirname(__file__),
                                                 'requirements.txt'))

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))
_version = find_version(os.path.join(os.path.dirname(__file__), "spacemerlin",
                                     "__init__.py"))

setup(
    name='spacemerlin',
    version=_version,
    packages=find_packages(),
    url='https://github.com/SpaceMeerkat/CAE',
    license='MIT',
    author='SpaceMeerkat',
    author_email='',
    test_suite="unittest",
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    tests_require=["coverage"],
    include_package_data=True
)
