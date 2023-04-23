from pathlib import Path

from setuptools import setup, find_packages


# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="pygpt4all",
    version="1.0.0",
    author="Abdeladim Sadiki",
    description="Official Python CPU inference for GPT4All language models based on llama.cpp and ggml",
    long_description=long_description,
    ext_modules=[],
    zip_safe=False,
    python_requires=">=3.8",
    packages=find_packages('.'),
    package_dir={'': '.'},
    long_description_content_type="text/markdown",
    license='MIT',
    # project_urls={
    #     'Documentation': '',
    #     'Source': '',
    #     'Tracker': '',
    # },
    install_requires=["pyllamacpp==1.0.6", "pygptj"],
    # extras_require={"all": [""]},
)
