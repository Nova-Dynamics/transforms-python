import re
from setuptools import setup, Extension

with open("robotransforms/_version.py") as f:
    exec(f.read())

try:
    with open("README.md") as f:
        readme = f.read()
except:
    readme = ""

def main():
    setup(
        name="robotransforms",
        version=__version__,
        description="A transformation library for robot motion", # TODO
        long_description=readme,
        long_description_content_type="text/markdown",
        author="Jonathan D. B. Van Schenck",
        author_email="jvschenck@novadynamics.com",
        license="MIT",
        url="https://github.com/Nova-Dynamics/transforms-python",
        packages=[
            "robotransforms",
            "robotransforms.utils",
            "robotransforms.dead_reckon",
            "robotransforms.euclidean",
        ],
        install_requires=[
            'numpy>=1.19',
        ],
        ext_modules=[
            Extension("utils_wrapper", [
                "robotransforms/utils/wrapper.cpp",
                "robotransforms/utils/base.cpp"
            ]),
            Extension("euclidean_wrapper", [
                "robotransforms/euclidean/wrapper.cpp",
                "robotransforms/euclidean/base.cpp",
            ]),
            Extension("dead_reckon_wrapper", [
                "robotransforms/dead_reckon/wrapper.cpp",
                "robotransforms/dead_reckon/base.cpp",
                "robotransforms/utils/base.cpp",
                "robotransforms/euclidean/base.cpp",
            ]),
        ]
    )

if __name__ == "__main__":
    main()
