from distutils.core import setup, Extension

with open("transforms/__version__.py") as f:
    exec(f.readline())

def main():
    setup(
        name="transforms",
        version=__version__,
        description="", # TODO
        author="Jonathan D. B. Van Schenck",
        author_email="jvschenck@novadynamics.com",
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
