from distutils.core import setup, Extension

with open("transforms/version.py") as f:
    exec(f.readline())

reqs = []
with open("requirements.txt") as f:
    for l in f:
        reqs.append(l.strip())

def main():
    setup(
        name="transforms",
        version=__version__,
        description="", # TODO
        author="Jonathan D. B. Van Schenck",
        author_email="jvschenck@novadynamics.com",
        install_requires=reqs,
        ext_modules=[
            Extension("utils_wrapper", [
                "transforms/utils/wrapper.cpp",
                "transforms/utils/base.cpp"
            ]),
            Extension("euclidean_wrapper", [
                "transforms/euclidean/wrapper.cpp",
                "transforms/euclidean/base.cpp"
            ]),
            Extension("dead_reckon_wrapper", [
                "transforms/dead_reckon/wrapper.cpp",
                "transforms/dead_reckon/base.cpp",
                "transforms/utils/base.cpp",
                "transforms/euclidean/base.cpp"
            ]),
        ]
    )

if __name__ == "__main__":
    main()
