from distutils.core import setup

setup(
	name='dekef',
	version='3.0',
	packages=['dekef'],
	url='https://github.com/zhoucx1119/dekef',
	license='MIT',
	author='Chenxi Zhou',
	author_email='zhoucx1989@gmail.com',
	description='Density estimation in kernel exponential families.',
	install_requires=['numpy', 'matplotlib', 'scipy']
)
