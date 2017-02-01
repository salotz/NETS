from setuptools import setup, find_packages

setup(
    name='NETS',
    version='0.1',
    py_modules=['nets'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'Numpy',
        'Pandas',
        'Scipy'
    ],
    entry_points={
        'console_scripts' : [
            "nets = nets.nets:cli"]}

)
